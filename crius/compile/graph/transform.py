#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph transformation. """

import os
from typing import (List, Callable, Any, Optional, Union, Tuple)

import torch
from torch import nn
import torch._dynamo as dynamo

from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan
from crius.global_envs import global_configs, logger

__all__ = [
    "transform_to_ir_graph",
    "propagate_node_shapes",
    "infer_node_layer_types",
    "insert_comm_primitives_into_graph",
    "construct_backward_graph",
]


def transform_to_ir_graph(
    model: nn.Module,
    *input_args,
) -> Tuple[IRGraph]:
    """ Transform model into IR graph, tracing with Python-level JIT compiler. 
    
    Args:
        model: Torch-like model to be compiled.
        input_args: Model input arguments.
    """

    fx_module = dynamo.export(model)(*input_args)[0]
    fx_graph = fx_module.graph
    # Construct IR graph
    graph = IRGraph(fx_graph.owning_module)
    output_vals = graph.graph_copy(fx_graph, val_map={}, return_output_node=True)
    output_val, old_output_node = output_vals
    graph.output(output_val, type_expr=getattr(old_output_node, "type", None))   # Insert output node
    return graph


def propagate_node_shapes(graph: Optional[IRGraph], *input_args) -> IRGraph:
    """ Propagate per-node input/output shapes in the IR graph. 
    
    Args:
        graph: IR graph to propagate shapes.
        input_args: Input arguments of the model `.forward()` for shape propagation.
    """
    if graph is None:
        return None
    
    from torch.fx.passes.shape_prop import ShapeProp
    
    ShapeProp(graph.owning_module).propagate(*input_args)
    graph = graph.owning_module.graph
    for node in graph.nodes:
        # Input shapes
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node) and hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                input_shapes.append(arg.meta['tensor_meta'].shape)
        setattr(node, 'input_shapes', input_shapes)
        
        # Output shapes
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            if not isinstance(tensor_meta, list):
                tensor_meta = [tensor_meta]
            output_shapes = [_meta.shape for _meta in tensor_meta if hasattr(_meta, 'shape')]
        else:
            output_shapes = []
        setattr(node, 'output_shapes', output_shapes)
    
    return graph


def infer_node_layer_types(graph: Optional[IRGraph]) -> IRGraph:
    """ Infer node layer types (e.g., gemm) in the IR graph. 

    NOTE(chunyu): The `node.target` should be strictly named, maybe add pre sanity check at the beginning.
    
    Args:
        graph: IR graph to infer layer types.
    """
    if graph is None:
        return None
    
    for node in graph.nodes:
        if hasattr(node, 'layer_type'):
            continue
        
        if node.op == 'call_module':
            if (
                # ------------- String matching rules -------------
                'fc' in str(node.target)
            ):
                setattr(node, 'layer_type', 'gemm')
            else:
                setattr(node, 'layer_type', 'unknown_call_module')
        
        elif node.op == 'call_function':
            if (
                # ------------- String matching rules -------------
                'all_reduce' in str(node.target)
            ):
                setattr(node, 'layer_type', 'all_reduce')
            elif (
                # ------------- String matching rules -------------
                'matmul' in str(node.target)
            ):
                setattr(node, 'layer_type', 'gemm')
            else:
                setattr(node, 'layer_type', 'unknown_call_function')
        
        else:
            setattr(node, 'layer_type', 'unknown_others')

    return graph


def insert_comm_primitives_into_graph(
    graph: IRGraph,
    parallelism_plan: ParallelismPlan,
    forward_pass: str,
 ) -> IRGraph:
    """ Insert IR nodes of communication primitives in IR graph, based on the parallelism plan. 
    
    Args:
        graph: IR graph to be inserted with communication primitives.
        parallelism_plan: How the to-be-compiled model would be parallelized.
        forward_pass: To insert communication operators in forward pass.
    """

    if (
        parallelism_plan is None or
        (
            parallelism_plan.pipeline_parallel_degree == 1 and
            parallelism_plan.tensor_parallel_degree == 1 and
            parallelism_plan.data_parallel_degree == 1
        ) or
        graph is None
    ):
        return graph    # No insertion needed
    
    return _insert_comm_primitives_into_graph_impl(graph, forward_pass)


def _insert_comm_primitives_into_graph_impl(graph: IRGraph, forward_pass: bool) -> IRGraph:
    """ Case-specific implementation for communication primitive insertion. 
    
    NOTE: Engineers should customize the `global_configs.comm_op_insert_rules`.
    """

    comm_op_insert_rules = global_configs.comm_op_insert_rules
    if comm_op_insert_rules is None:
        logger.warning("No communication operator insertion rules are specified in `global_configs`.")
        return graph
    
    for comm_primitive_str in comm_op_insert_rules.keys():
        comm_primitive = comm_primitive_str.split(":")[0]   # Remove parallelism suffix
        if comm_primitive not in global_configs.supported_comm_primitives:
            logger.warning(f"Unsupported communication primitive `{comm_primitive}`.")
            continue
        
        # Traverse the node to insert
        after_nodes = comm_op_insert_rules[comm_primitive_str].get("after_nodes", [])
        interval = comm_op_insert_rules[comm_primitive_str].get("interval", 1)
        node_pass = comm_op_insert_rules[comm_primitive_str].get("pass", "forward")
        node_parallelism = comm_op_insert_rules[comm_primitive_str].get("parallelism", "tp")
        after_node_num = 0

        if forward_pass:
            for node in graph.nodes:
                for after_node_keyword in after_nodes:
                    if after_node_keyword not in str(node.target):
                        continue

                    after_node_num += 1
                    if after_node_num % interval != 0:
                        continue    # Should not insert here

                    if node_pass != "forward":
                        continue

                    # Check whether the graph already has the comm op node
                    if comm_primitive in str(node.next.target):
                        logger.info(f"=======> Node `{str(node.next.target)}` already exists after node `{str(node.target)}`, skip insertion.")
                        setattr(node.next, "node_pass", node_pass)
                        setattr(node.next, "node_parallelism", node_parallelism)
                        continue
                    
                    # TODO(chunyu): When manually inserting communication nodes, we may:
                    # 1. Construct a communication node;
                    # 2. Determine its input shape based on its prev node's output shape;
                    # 3. Maybe need to change shapes of next nodes (e.g., when comm primitive is all-gather);
                    raise NotImplementedError("Implement customized communication node insertion here.")
        
        else:
            for node in graph.nodes:
                assert node.layer_type in ("gemm", ), f"Invalid node layer type in backward graph: {node.layer_type}"
                for after_node_keyword in after_nodes:
                    if after_node_keyword not in str(node.target):
                        continue

                    after_node_num += 1
                    if after_node_num % interval != 0:
                        continue    # Should not insert here

                    if node_pass != "backward":
                        continue
                    
                    assert comm_primitive not in str(node.next.target), \
                        "In manually constructed backward graph, no communication operator should be included at first."
                    
                    # Insert a communication node
                    with graph.inserting_after(node):
                        comm_node = graph.create_node(op="call_function", target=comm_primitive)
                    
                    raise NotImplementedError("This is wrong, fix communication node insertion...")
    
            for node in graph.nodes:
                print(node.layer_type, node.target)
            
            exit(0)

    return graph


def construct_backward_graph(graph: IRGraph, mode: str = "manual") -> IRGraph:
    """ Construct backward IR graph with only GEMM and communication operators. 
    
    Args:
        graph: Forward IR graph that has been shape-propogated and inserted communication operators.
        mode: Mode for constructing backward IR graph. Set to 'manual' means we manually construct graph by traversing 
              the forward IR graph; TODO(chunyu): Maybe use dynamo to trace the backward graph.
    """

    assert mode == "manual", f"Currently only support manually constructing backward IR graph, got mode: {mode}"
    if not global_configs.enable_backward_graph_generation:
        return None
    
    bw_graph = IRGraph()
    for node in reversed(graph.nodes):
        if node.layer_type == "gemm":   # [b, s, h1] x [h1, h2] -> [b, s, h2]
            assert len(node.input_shapes[0]) == len(node.output_shapes[0]) == 3
            assert len(node.input_shapes[1]) == 2
            (batch_size, seq_len, left_hidden_dim) = node.input_shapes[0]
            right_hidden_dim = node.input_shapes[1][-1]

            # Input gradient: [b, s, h2] x [h2, h1] -> [b, s, h1]
            bw_node_input_grad = bw_graph.create_node(op="call_function", target=node.target)
            input_shapes_ = [torch.Size([batch_size, seq_len, right_hidden_dim]), torch.Size([right_hidden_dim, left_hidden_dim])]
            output_shapes_ = [torch.Size([batch_size, seq_len, left_hidden_dim])]
            setattr(bw_node_input_grad, 'input_shapes', input_shapes_)
            setattr(bw_node_input_grad, 'output_shapes', output_shapes_)
            setattr(bw_node_input_grad, 'layer_type', 'gemm')

            # Weight gradient: [b, h1, s] x [b, s, h2] -> [bs, h1, h2] -> [h1, h2]
            bw_node_weight_grad = bw_graph.create_node(op="call_function", target=node.target)
            input_shapes_ = [torch.Size([batch_size, left_hidden_dim, seq_len]), torch.Size([batch_size, seq_len, right_hidden_dim])]
            output_shapes_ = [torch.Size([batch_size, left_hidden_dim, right_hidden_dim])]
            setattr(bw_node_weight_grad, 'input_shapes', input_shapes_)
            setattr(bw_node_weight_grad, 'output_shapes', output_shapes_)
            setattr(bw_node_weight_grad, 'layer_type', 'gemm')

            # FIXME: Currently we ignore the sum operation (dim0) of weight gradient...

        elif node.layer_type == "all_reduce":
            pass    # Directly return out_grad, which are the same across all ranks, as outputs are the same after allreduce in forward
        else:
            pass    # Other non-dominant operators

    return bw_graph
