#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph transformation. """

import os
from typing import (List, Callable, Any, Optional, Union)

import torch
from torch import nn
import torch._dynamo as dynamo
from torch.fx import Graph, Node

from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan

__all__ = [
    "transform_to_ir_graph",
    "propagate_node_shapes",
    "infer_node_layer_types",
    "insert_comm_primitives_into_graph",
]


def transform_to_ir_graph(
    model: nn.Module,
    *input_args,
) -> Any:
    """ Transform model into IR graph. """
    # Trace with Python-level JIT compiler
    fx_module = dynamo.export(model)(*input_args)[0]
    fx_graph = fx_module.graph
    # Construct IR graph
    graph = IRGraph(fx_graph.owning_module)
    output_vals = graph.graph_copy(fx_graph, val_map={}, return_output_node=True)
    output_val, old_output_node = output_vals
    graph.output(output_val, type_expr=getattr(old_output_node, "type", None))   # Insert output node
    return graph


def propagate_node_shapes(graph: IRGraph, *input_args) -> IRGraph:
    """ Propagate per-node input/output shapes in the IR graph. 
    
    Args:
        graph: IR graph to propagate shapes.
        input_args: Input arguments of the model `.forward()` for shape propagation.
    """
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


def infer_node_layer_types(graph: IRGraph) -> IRGraph:
    """ Infer node layer types (e.g., gemm) in the IR graph. 

    NOTE(chunyu): The `node.target` should be strictly named, maybe add pre sanity check at the beginning.
    
    Args:
        graph: IR graph to infer layer types.
    """
    for node in graph.nodes:
        if hasattr(node, 'layer_type'):
            continue
        
        # GEMM
        if node.op == 'call_module':
            if (
                # ------------- String matching rules -------------
                'fc' in node.target
            ):
                setattr(node, 'layer_type', 'gemm')
            else:
                setattr(node, 'layer_type', 'unknown')

    return graph


def insert_comm_primitives_into_graph(
    # graph: IRGraph,
    graph: Graph,
    parallelism_plan: ParallelismPlan,
 ) -> IRGraph:
    """ Insert IR nodes of communication primitives in IR graph, based on the parallelism plan. 
    
    Args:
        graph: IR graph to be inserted with communication primitives.
        parallelism_plan: How the to-be-compiled model would be parallelized.
    """
    # TODO(chunyu): Implement this function.
    def is_column_parallel_linear_boundary(node: Node, graph: Graph) -> bool:
        if node.name in graph.node_layer_mapping:
            node_layer_info = graph.node_layer_mapping[node.name]
        else:
            return False
        
        if str(node_layer_info['layer_class']) != "ColumnParallelLinear":
            return False
        
        next_node = node.next

        if next_node is None:
            return True
        
        next_node_layer_info = graph.node_layer_mapping[next_node.name]

        if str(next_node_layer_info['layer_class']) != "ColumnParallelLinear" or \
        (str(next_node_layer_info['layer_class']) == "ColumnParallelLinear" and \
         str(node_layer_info['module_path']) != str(next_node_layer_info['module_path'])):
            return True
        
        return False
    
    def is_row_parallel_linear_boundary(node: Node, graph: Graph) -> bool:
        if node.name in graph.node_layer_mapping:
            node_layer_info = graph.node_layer_mapping[node.name]
        else:
            return False
        
        if str(node_layer_info['layer_class']) != "RowParallelLinear":
            return False
        
        next_node = node.next

        if next_node is None:
            return True
        
        next_node_layer_info = graph.node_layer_mapping[next_node.name]
        
        if str(next_node_layer_info['layer_class']) != "RowParallelLinear" or \
        (str(next_node_layer_info['layer_class']) == "RowParallelLinear" and \
         str(node_layer_info['module_path']) != str(next_node_layer_info['module_path'])):
            return True
        
        return False
    
    # def is_pipeline_boundary(node: Node, graph: Graph) -> bool:
    #     def _is_forward_step_boundary(node: torch.fx.Node, graph: fx.Graph) -> bool:
    #         if hasattr(node, 'target') and isinstance(node.target, str):
    #             if 'forward_step' in node.target:
    #                 return True
                
    #             for user in graph.users[node]:
    #                 if hasattr(user, 'target') and 'forward_step' in str(user.target):
    #                     return True
            
    #         return False

    #     def _involves_pipeline_parallel_rank(node: torch.fx.Node) -> bool:
    #         pipeline_rank_patterns = [
    #             'get_pipeline_model_parallel_rank',
    #             'pipeline_parallel_rank',
    #             'pp_rank',
    #             'stage_id'
    #         ]
            
    #         if hasattr(node, 'target') and isinstance(node.target, str):
    #             target_str = str(node.target).lower()
    #             if any(pattern in target_str for pattern in pipeline_rank_patterns):
    #                 return True
            
    #         for arg in node.args:
    #             if isinstance(arg, str) and any(pattern in arg.lower() for pattern in pipeline_rank_patterns):
    #                 return True
            
    #         return False
        

    #     if _is_forward_step_boundary(node, graph):
    #         return True
        
    #     if _involves_pipeline_parallel_rank(node):
    #         return True
        
    #     return False
        
    # tp_group = parallelism_plan.tensor_parallel_group
    # tp_size = parallelism_plan.tensor_parallel_degree

    # if tp_size <= 1:
    #     return graph
    
    # comm_ops = {
    #     'all_gather': (torch.distribution.all_gather, {'group': tp_group}),
    #     'all_reduce': (torch.distribution.all_reduce, {'group': tp_group, 'op': torch.distribution.ReduceOp.SUM})
    # }

    for node in graph.nodes:
        # print(f"is_cb:{is_column_parallel_linear_boundary(node,graph)}, is_rb:{is_row_parallel_linear_boundary(node,graph)}")
        if is_column_parallel_linear_boundary(node, graph):
            with graph.inserting_after(node):
                comm_node = graph.call_function(torch.distributed.all_gather)
                # comm_node = Node(graph=graph, name="all_gather", op="call_function", target=torch.distributed.all_gather, args=[], kwargs={})
        if is_row_parallel_linear_boundary(node, graph):
            with graph.inserting_after(node):
                comm_node = graph.call_function(torch.distributed.all_reduce)
                # comm_node = Node(graph=graph, name="all_reduce", op="call_function", target=torch.distributed.all_gather, args=[], kwargs={})

        # if is_pipeline_boundary(node, graph):
        #     with graph.inserting_after(node):
        #         comm_node = graph.call_function(torch.distributed.send_rev)
            

    # print("\n")
    # print(graph)
    return graph

