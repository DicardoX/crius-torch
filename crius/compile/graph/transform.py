#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph transformation. """

import os
from typing import (List, Callable, Any, Optional, Union)

import torch
from torch import nn
import torch._dynamo as dynamo

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
    graph: IRGraph,
    parallelism_plan: ParallelismPlan,
 ) -> IRGraph:
    """ Insert IR nodes of communication primitives in IR graph, based on the parallelism plan. 
    
    Args:
        graph: IR graph to be inserted with communication primitives.
        parallelism_plan: How the to-be-compiled model would be parallelized.
    """
    # TODO(chunyu): Implement this function.
    return graph
