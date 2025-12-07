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
    "shard_compute_node_input_output_shapes",
    "insert_comm_primitives_into_graph",
    "construct_backward_graph",
]


def transform_to_ir_graph(model: nn.Module, *input_args, **input_kwargs) -> Tuple[IRGraph]:
    """ Transform model into IR graph, tracing with Python-level JIT compiler. 
    
    Args:
        model: Torch-like model to be compiled.
        input_args: Model input arguments.
        input_kwargs: Model input keyword arguments.
    """

    fx_module = dynamo.export(model)(*input_args, **input_kwargs)[0]
    fx_graph = fx_module.graph
    # Construct IR graph
    graph = IRGraph(fx_graph.owning_module)
    output_vals = graph.graph_copy(fx_graph, val_map={}, return_output_node=True)
    output_val, old_output_node = output_vals
    graph.output(output_val, type_expr=getattr(old_output_node, "type", None))   # Insert output node
    return graph


def propagate_node_shapes(graph: Optional[IRGraph], *input_args, **input_kwargs) -> IRGraph:
    """ Propagate per-node input/output shapes in the IR graph. 
    
    Args:
        graph: IR graph to propagate shapes.
        input_args: Input arguments of the model `.forward()` for shape propagation.
        input_kwargs: Input keyword arguments of the model `.forward()` for shape propagation.
    """
    if graph is None:
        return None
    
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
    
    # .propagate() only takes positional args
    all_args = (*input_args, *input_kwargs.values())

    # assert str(global_configs.current_device) != "cpu", \
    #     "Shape propagation currently hangs in CPU mode, or prabably due to extremely slow progagation."
    ShapeProp(graph.owning_module).propagate(*all_args)
    graph_ = graph.owning_module.graph
    for i, node in enumerate(graph_.nodes):
        # Input shapes
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node) and hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                in_meta = arg.meta['tensor_meta']
                if not isinstance(in_meta, TensorMetadata):
                    # Special operators (e.g., getitem), non-negligible latency, thus we directly use the first tensor's shape
                    logger.warning(f"Node {node.target} has multiple tensor metas, use the first one for input shape.")
                    in_meta = in_meta[0]       
                
                input_shapes.append(in_meta.shape)
        # setattr(node, 'input_shapes', input_shapes)
        setattr(list(graph.nodes)[i], 'input_shapes', input_shapes)
        
        # Output shapes
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            out_meta = node.meta['tensor_meta']
            if not isinstance(out_meta, list):
                out_meta = [out_meta]
            output_shapes = [_meta.shape for _meta in out_meta if hasattr(_meta, 'shape')]
        else:
            output_shapes = []
        # setattr(node, 'output_shapes', output_shapes)
        setattr(list(graph.nodes)[i], 'output_shapes', output_shapes)

        # Data type
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta'] \
                if isinstance(node.meta['tensor_meta'], TensorMetadata) else node.meta['tensor_meta'][0]
            dtype = tensor_meta.dtype
        else:
            dtype = None
        setattr(list(graph.nodes)[i], 'dtype', dtype)

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
        setattr(node.next, "node_pass", "forward")

        if getattr(node, "layer_type", None) is not None:
            continue
        
        if node.op == "call_module":
            if (
                # ------------- String matching rules -------------
                "fc" in str(node.target)
            ):
                setattr(node, "layer_type", "gemm")
                setattr(node, "layout", "s:b:h1,h1:h2")     # Used in Megatron
            elif (
                # ------------- String matching rules -------------
                "layernorm" in str(node.target)
            ):
                setattr(node, "layer_type", "layernorm")
                setattr(node, "layout", "s:b:h1")
            else:
                setattr(node, "layer_type", "unknown_call_module")
                setattr(node, "layout", None)
        
        elif node.op == "call_function":
            if (
                # ------------- String matching rules -------------
                "all_reduce" in str(node.target)
            ):
                setattr(node, "layer_type", "all_reduce")
                setattr(node, "layout", None)
            elif (
                # ------------- String matching rules -------------
                "matmul" in str(node.target)
            ):
                setattr(node, "layer_type", "gemm")
                setattr(node, "layout", "s:b:h1,h1:h2")     # Used in Megatron
            elif (
                # ------------- String matching rules -------------
                "baddbmm" in str(node.target)
            ):
                setattr(node, "layer_type", "baddbmm")
                setattr(node, "layout", "b:s:s,b:s:h1,b:h1:s")  # e.g., Q @ K^T + attn mask
            elif (
                # ------------- String matching rules -------------
                "bmm" in str(node.target) and "baddbmm" not in str(node.target)
            ):
                setattr(node, "layer_type", "bmm")
                setattr(node, "layout", "b:s:s,b:s:h1")     # e.g., attn score @ V
            else:
                setattr(node, "layer_type", "unknown_call_function")
                setattr(node, "layout", None)
        
        else:
            setattr(node, "layer_type", "unknown_others")
            setattr(node, "layout", None)

    return graph


def shard_compute_node_input_output_shapes(
    graph: IRGraph, 
    parallelism_plan: ParallelismPlan,
) -> IRGraph:
    """ Shard compute nodes' input and output shapes (e.g., QKV proj) based on the parallelism 
    plan, currently only consider sharding dominant (e.g., GEMM) nodes. 

    Notably, this function is only used for the tuner, since Megatron's global world sizes 
    (e.g., TP) are not set -> weight shapes of model's Column/RowParallelLinear are not sharded.

    Args:
        graph: IR graph to be sharded.
        parallelism_plan: How the to-be-compiled model would be parallelized.
    """

    # All dominant nodes to be sharded
    dominant_node_infos_tp = [
        "self_attention_linear_qkv_weight",
        "self_attention_linear_proj_weight",
        "mlp_linear_fc1_weight",
        "mlp_linear_fc2_weight",
        "self_attention_core_attention_attention_dropout",
        "output_layer_weight",
    ]
    tp_size = parallelism_plan.tensor_parallel_degree

    cur_marker, cur_marker_keyword = None, None
    for node in graph.nodes:
        if "L__self___module" in str(node.target):
            cur_marker = str(node.target)
            cur_marker_keyword = None   # Reset
            for kw in dominant_node_infos_tp:
                if kw in cur_marker:
                    cur_marker_keyword = kw
                    break

        if cur_marker_keyword in (
            "self_attention_linear_qkv_weight", "mlp_linear_fc1_weight", "output_layer_weight", 
        ):
            if node.layer_type == "gemm":   # [s, b, h1] x [h1, h2] -> [s, b, h2]
                # Column parallel linear
                assert len(node.input_shapes) == 2
                assert len(node.output_shapes) == 1
                (s, b, h1) = node.input_shapes[0]
                h2 = node.input_shapes[1][-1]
                node.input_shapes[1] = torch.Size([h1, h2 // tp_size])
                node.output_shapes[0] = torch.Size([s, b, h2 // tp_size])
            
            elif node.layer_type == "baddbmm":    
                # [b * num_heads, s, s] + [b * num_heads, s, head_dim] x [b * num_heads, head_dim, s]
                assert len(node.input_shapes) == 3
                assert len(node.output_shapes) == 1
                (b, s, _) = node.input_shapes[0]    # Here, `b` denotes batch size * num heads
                h1 = node.input_shapes[1][-1]       # Here, `h1` denotes head dimention
                node.input_shapes[0] = torch.Size([b // tp_size, s, s])
                node.input_shapes[1] = torch.Size([b // tp_size, s, h1])
                node.input_shapes[2] = torch.Size([b // tp_size, h1, s])
                node.output_shapes[0] = torch.Size([b // tp_size, s, s])
        
        elif cur_marker_keyword == "self_attention_core_attention_attention_dropout":
            if node.layer_type == "bmm":
                # [b * num_heads, s, s] x [b * num_heads, s, head_dim]
                assert len(node.input_shapes) == 2
                assert len(node.output_shapes) == 1
                (b, s, _) = node.input_shapes[0]    # Here, `b` denotes batch size * num heads
                h1 = node.input_shapes[1][-1]       # Here, `h1` denotes head dimention
                node.input_shapes[0] = torch.Size([b // tp_size, s, s])
                node.input_shapes[1] = torch.Size([b // tp_size, s, h1])
                node.output_shapes[0] = torch.Size([b // tp_size, s, h1])                

        elif cur_marker_keyword in ("self_attention_linear_proj_weight", "mlp_linear_fc2_weight", ):
            if node.layer_type == "gemm": 
                # Row parallel linear
                assert len(node.input_shapes) == 2
                assert len(node.output_shapes) == 1
                (s, b, h1) = node.input_shapes[0]
                h2 = node.input_shapes[1][-1]
                node.input_shapes[0] = torch.Size([s, b, h1 // tp_size])
                node.input_shapes[1] = torch.Size([h1 // tp_size, h2])
    
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
    
    return _insert_comm_primitives_into_graph_impl(graph, parallelism_plan, forward_pass)


def _insert_comm_primitives_into_graph_impl(
    graph: IRGraph, parallelism_plan: ParallelismPlan, forward_pass: bool,
) -> IRGraph:
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
        after_markers = comm_op_insert_rules[comm_primitive_str].get("after_markers", [])
        after_nodes = comm_op_insert_rules[comm_primitive_str].get("after_nodes", [])
        skip_first_n = comm_op_insert_rules[comm_primitive_str].get("skip_first_n", 0)
        interval = comm_op_insert_rules[comm_primitive_str].get("interval", 1)
        node_pass = comm_op_insert_rules[comm_primitive_str].get("pass", "forward")
        node_parallelism = comm_op_insert_rules[comm_primitive_str].get("parallelism", None)
        
        n_ranks = 1
        if node_parallelism == "tp":
            n_ranks = parallelism_plan.tensor_parallel_degree
        elif node_parallelism == "dp":
            n_ranks = parallelism_plan.data_parallel_degree
        else:
            raise ValueError(f"Unsupported node parallelism: {node_parallelism}")
        
        after_node_num = 0
        if forward_pass:
            # Forward pass
            cur_marker = None
            for node in graph.nodes:
                if "L__self___module" in str(node.target):
                    cur_marker = str(node.target)                

                for after_node_keyword in after_nodes:
                    if after_node_keyword not in str(node.target):
                        continue    # Node target not matched

                    after_node_num += 1
                    if (
                        # Skip the first n matched nodes
                        after_node_num <= skip_first_n or
                        # We insert every `interval` matched nodes
                        (after_node_num - skip_first_n - 1) % interval != 0 or
                        node_pass != "forward" or
                        # No parallelism
                        n_ranks <= 1
                    ):
                        continue

                    if all([_marker not in cur_marker for _marker in after_markers]):
                        continue    # Not belong to any specified marker

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
            # Backward pass
            for node in graph.nodes:
                if node.layer_type == comm_primitive:
                    continue    # Just inserted

                assert node.layer_type in ("gemm", "bmm", "baddbmm", ), f"Invalid node layer type in backward graph: {node.layer_type}"
                for after_node_keyword in after_nodes:
                    if after_node_keyword not in str(node.target):
                        continue

                    after_node_num += 1
                    if (
                        # Skip the first n matched nodes
                        after_node_num <= skip_first_n or
                        # We insert every `interval` matched nodes
                        (after_node_num - skip_first_n - 1) % interval != 0 or
                        node_pass != "backward" or
                        # No parallelism
                        n_ranks <= 1
                    ):
                        continue
                    
                    # assert comm_primitive not in str(node.next.target), \
                        # "In manually constructed backward graph, no communication operator should be included at first."
                    
                    # Insert a communication node
                    with graph.inserting_after(node):
                        comm_node = graph.call_function(global_configs.comm_op_funcs[comm_primitive], args=(node, ))
                        setattr(comm_node, "layer_type", comm_primitive)
                        input_shapes, output_shapes = _infer_comm_node_input_output_shapes(node, comm_primitive, node_pass, node_parallelism)
                        setattr(comm_node, "input_shapes", input_shapes)
                        setattr(comm_node, "output_shapes", output_shapes)
                        setattr(comm_node, "node_pass", node_pass)
                        setattr(comm_node, "node_parallelism", node_parallelism)
                        setattr(comm_node, "fw_node", node.fw_node)   # Link to forward node

    return graph


def _infer_comm_node_input_output_shapes(
    prev_node: IRNode, primitive_type: str, node_pass: str, node_parallelism: str,
) -> Tuple[List[Any], List[Any]]:
    """ Case-specific infer the input/output shapes of a communication node. 
    
    TODO(chunyu): For shape-variable primitives (e.g., all-gather), we should consider parallel degree here.
    For example, when all-gather model parameters in forward pass for FSDP, the output shapes should be 
    n_ranks * input_shapes.
    """

    if primitive_type == "all_reduce" and node_pass == "backward" and node_parallelism == "dp":
        # All-reduce the weight gradients (output) of previous node
        assert len(prev_node.output_shapes[0]) == 2     # [s*b, h2] x [h2, h1]
        input_shapes = output_shapes = [prev_node.output_shapes[0]]
    elif primitive_type == "all_reduce" and node_pass == "backward" and node_parallelism == "tp":
        # All-reduce the activation gradients (input) of previous node
        assert len(prev_node.input_shapes[1]) == 2         # [s*b, h2] x [h2, h1]
        input_shapes = output_shapes = [prev_node.input_shapes[0]]
    else:
        raise NotImplementedError("Unsupport case when inferring input/output shapes of communication node.")
    
    return input_shapes, output_shapes


def construct_backward_graph(graph: IRGraph, mode: str = "manual") -> IRGraph:
    """ Construct backward IR graph with only heavy operators (e.g., GEMM) and communication operators. 
    
    Args:
        graph: Forward IR graph that has been shape-propogated and inserted communication operators.
        mode: Mode for constructing backward IR graph. Set to 'manual' means we manually construct graph by traversing 
              the forward IR graph; TODO(chunyu): Maybe use dynamo to trace the backward graph.
    """

    assert mode == "manual", f"Currently only support manually constructing backward IR graph, got mode: {mode}"
    if global_configs.only_forward or not global_configs.enable_backward_graph_generation:
        return None
    
    bw_graph = IRGraph()
    for node in reversed(graph.nodes):
        if node.layer_type == "gemm":   # [s, b, h1] x [h1, h2] -> [s, b, h2]
            assert len(node.input_shapes[0]) == len(node.output_shapes[0]) == 3
            assert len(node.input_shapes[1]) == 2
            (batch_size, seq_len, left_hidden_dim) = node.input_shapes[0]
            right_hidden_dim = node.input_shapes[1][-1]

            # Input gradient: [s, b, h2] x [h2, h1] -> [s, b, h1]
            def __fake_bw_func_gemm_input_grad():
                return None

            bw_node_input_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_gemm_input_grad)
            input_shapes_ = [torch.Size([batch_size, seq_len, right_hidden_dim]), torch.Size([right_hidden_dim, left_hidden_dim])]
            output_shapes_ = [torch.Size([batch_size, seq_len, left_hidden_dim])]
            setattr(bw_node_input_grad, "node_pass", "backward")
            setattr(bw_node_input_grad, "input_shapes", input_shapes_)
            setattr(bw_node_input_grad, "output_shapes", output_shapes_)
            setattr(bw_node_input_grad, "layer_type", "gemm")
            setattr(bw_node_input_grad, "layout", "s:b:h2,h2:h1")
            setattr(bw_node_input_grad, "fw_node", node)

            # Weight gradient: grad_weight = input^T @ grad_output
            # Intuitively, the shapes should be: 
            #                   [s, h1, b] x [s, b, h2] -> [s, h1, h2]
            # To avoid extremely large intermediate tensor [s, h1, h2], the common practice is to merge s and b
            # dimensions, transforming input and grad_output into 2D tensors. Therefore, the shapes become: 
            #                   [h1, s*b] x [s*b, h2] -> [h1, h2]
            # For example, Megatron performs such transformation in 
            # `LinearWithGradAccumulationAndAsyncCommunication` (`prepare_input_tensors_for_wgrad_compute()` 
            # in megatron/core/utils.py):
            # 
            #       # Convert the tensor shapes to 2D for execution compatibility
            #       if grad_output.dim() == 3:
            #           grad_output = grad_output.view(
            #               grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            #           )
            #           all_gathered_input = all_gathered_input.view(
            #               all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
            #           )
            def __fake_bw_func_gemm_weight_grad():
                return None
            
            bw_node_weight_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_gemm_weight_grad)
            input_shapes_ = [torch.Size([left_hidden_dim, seq_len*batch_size]), torch.Size([seq_len*batch_size, right_hidden_dim])]
            output_shapes_ = [torch.Size([left_hidden_dim, right_hidden_dim])]
            setattr(bw_node_weight_grad, "node_pass", "backward")
            setattr(bw_node_weight_grad, "input_shapes", input_shapes_)
            setattr(bw_node_weight_grad, "output_shapes", output_shapes_)
            setattr(bw_node_weight_grad, "layer_type", "gemm")
            setattr(bw_node_weight_grad, "layout", "h1:s*b,s*b:h2")
            setattr(bw_node_weight_grad, "fw_node", node)

        elif node.layer_type == "bmm":  # [b, s, s] x [b, s, h1] -> [b, s, h1]
            assert len(node.input_shapes[0]) == len(node.input_shapes[1]) == len(node.output_shapes[0]) == 3
            (batch_size, seq_len, left_hidden_dim) = node.input_shapes[1]
            
            # Attn score gradient: [b, s, h1] x [b, h1, s] -> [b, s, s]
            def __fake_bw_func_bmm_attn_score_grad():
                return None
            
            bw_node_attn_score_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_bmm_attn_score_grad)
            input_shapes_ = [torch.Size([batch_size, seq_len, left_hidden_dim]), torch.Size([batch_size, left_hidden_dim, seq_len])]
            output_shapes_ = [torch.Size([batch_size, seq_len, seq_len])]
            setattr(bw_node_attn_score_grad, "node_pass", "backward")
            setattr(bw_node_attn_score_grad, "input_shapes", input_shapes_)
            setattr(bw_node_attn_score_grad, "output_shapes", output_shapes_)
            setattr(bw_node_attn_score_grad, "layer_type", "bmm")
            setattr(bw_node_attn_score_grad, "layout", "b:s:h1,b:h1:s")
            setattr(bw_node_attn_score_grad, "fw_node", node)

            # Value tensor gradient: [b, s, s] x [b, s, h1] -> [b, s, h1]
            def __fake_bw_func_bmm_value_tensor_grad():
                return None
            
            bw_node_value_tensor_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_bmm_value_tensor_grad)
            input_shapes_ = [torch.Size([batch_size, seq_len, seq_len]), torch.Size([batch_size, seq_len, left_hidden_dim])]
            output_shapes_ = [torch.Size([batch_size, seq_len, left_hidden_dim])]
            setattr(bw_node_value_tensor_grad, "node_pass", "backward")
            setattr(bw_node_value_tensor_grad, "input_shapes", input_shapes_)
            setattr(bw_node_value_tensor_grad, "output_shapes", output_shapes_)
            setattr(bw_node_value_tensor_grad, "layer_type", "bmm")
            setattr(bw_node_value_tensor_grad, "layout", "b:s:s,b:s:h1")
            setattr(bw_node_value_tensor_grad, "fw_node", node)
        
        elif node.layer_type == "baddbmm":  # [b, s, s] + [b, s, h1] x [b, h1, s] -> [b, s, s]
            assert len(node.input_shapes) == 3
            assert len(node.input_shapes[0]) == len(node.input_shapes[1]) == len(node.input_shapes[2]) == len(node.output_shapes[0]) == 3
            (batch_size, seq_len, left_hidden_dim) = node.input_shapes[1]

            # Query tensor gradient: [b, s, s] x [b, s, h1] -> [b, s, h1]
            def __fake_bw_func_baddbmm_query_tensor_grad():
                return None
            
            bw_node_query_tensor_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_baddbmm_query_tensor_grad)
            input_shapes_ = [torch.Size([batch_size, seq_len, seq_len]), torch.Size([batch_size, seq_len, left_hidden_dim])]
            output_shapes_ = [torch.Size([batch_size, seq_len, left_hidden_dim])]
            setattr(bw_node_query_tensor_grad, "node_pass", "backward")
            setattr(bw_node_query_tensor_grad, "input_shapes", input_shapes_)
            setattr(bw_node_query_tensor_grad, "output_shapes", output_shapes_)
            setattr(bw_node_query_tensor_grad, "layer_type", "bmm")                 # Backward pass of baddbmm involves bmm
            setattr(bw_node_query_tensor_grad, "layout", "b:s:s,b:s:h1")
            setattr(bw_node_query_tensor_grad, "fw_node", node)

            # Key tensor gradient: [b, h1, s] x [b, s, s] -> [b, h1, s]
            def __fake_bw_func_baddbmm_key_tensor_grad():
                return None
            
            bw_node_key_tensor_grad = bw_graph.create_node(op="call_function", target=__fake_bw_func_baddbmm_key_tensor_grad)
            input_shapes_ = [torch.Size([batch_size, left_hidden_dim, seq_len]), torch.Size([batch_size, seq_len, seq_len])]
            output_shapes_ = [torch.Size([batch_size, left_hidden_dim, seq_len])]
            setattr(bw_node_key_tensor_grad, "node_pass", "backward")
            setattr(bw_node_key_tensor_grad, "input_shapes", input_shapes_)
            setattr(bw_node_key_tensor_grad, "output_shapes", output_shapes_)
            setattr(bw_node_key_tensor_grad, "layer_type", "bmm")
            setattr(bw_node_key_tensor_grad, "layout", "b:h1:s,b:s:s")
            setattr(bw_node_key_tensor_grad, "fw_node", node)

        elif node.layer_type == "all_reduce":
            pass    # Directly return out_grad, which are the same across all ranks, as outputs are the same after allreduce in forward
        else:
            pass    # Other non-dominant operators

    return bw_graph
