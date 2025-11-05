#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Implementation of Torch-like model compiler. """

from typing import (Any, List, Tuple, Callable, Optional, Dict)
from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.fx import GraphModule

from crius.compile.graph.transform import (
    transform_to_ir_graph, propagate_node_shapes, infer_node_layer_types, insert_comm_primitives_into_graph,
    construct_backward_graph)
from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan

__all__ = [
    "CompileConfigs",
    "precompile",
    "postcompile",
    "compile",
]


@dataclass
class CompileConfigs:
    """ Compilation configurations. """
    
    base_input_args: Tuple[Any]
    """ Input arguments of model `.forward()` for IR graph transformation. """

    base_input_kwargs: Dict[str, Any]
    """ Input keyword arguments of model `.forward()` for IR graph transformation. """

    parallelism_plan: ParallelismPlan
    """ Parallelism plan of the model. """


def precompile(
    model: nn.Module, 
    base_input_args: Tuple[Any], 
    base_input_kwargs: Dict[str, Any],
) -> IRGraph:
    """ Pre-compilation with (1) IR graph transformation and (2) Shape propagation.
    
    This is used in Crius's planner, where multiple IR graphs should be compiled. 
    `dynamo.export()` would cause compilation error if repetatively called: 
        torch._dynamo.exc.UserError: Cannot export model which references tensors that are 
        neither buffers/parameters/constants nor are direct inputs.  For each tensor, if you'd 
        like this tensor to be an explicit input, add it as a dummy argument to the top-level 
        model definition you are exporting; if you would like its value to be embedded as an 
        exported constant, wrap its access in a function marked with @assume_constant_result.

    Args:
        model: Torch-like model to be compiled.
        base_input_args: (Optional) Input arguments of model `.forward()` for IR graph transformation.
        base_input_kwargs: (Optional) Input keyword arguments of model `.forward()` for IR graph transformation.
    """

    # Step 1. Transform model into IR graph
    graph = transform_to_ir_graph(model, *base_input_args, **base_input_kwargs)
    # Step 2. Propagate per-node input/output shapes
    graph = propagate_node_shapes(graph, *base_input_args, **base_input_kwargs)
    graph = infer_node_layer_types(graph)
    return graph


def postcompile(
    precompiled_graph: IRGraph,
    parallelism_plan: ParallelismPlan,
) -> List[IRGraph]: 
    """ Post-compilation with (1) Communication insertion and (2) Backward graph construction. 
    
    This is also used in Crius's planner, which is parallelism-specific and repeatly called.
    
    Args:
        precompiled_graph: Forward IR graph.
        parallelism_plan: The parallelism plan for the model to be profiled with.
    """

    # Step 3. Insert communication primitives
    # NOTE: The deepcopy is necessary to maintain independent graphs for different parallelism plans
    graph = insert_comm_primitives_into_graph(deepcopy(precompiled_graph), parallelism_plan, forward_pass=True)
    # Step 4. Construct backward graph
    bw_graph = construct_backward_graph(graph)
    bw_graph = insert_comm_primitives_into_graph(bw_graph, parallelism_plan, forward_pass=False)
    return graph, bw_graph


def compile(
    model: nn.Module,
    compile_configs: CompileConfigs,
) -> IRGraph:
    """ Python-to-IR compilation. 
    
    Args:
        model: Torch-like model to be compiled.
        compile_configs: Compilation configurations.
    """
    
    # Step 1. Transform model into IR graph
    graph = transform_to_ir_graph(model, *compile_configs.base_input_args, **compile_configs.base_input_kwargs)
    # Step 2. Propagate per-node input/output shapes
    graph = propagate_node_shapes(graph, *compile_configs.base_input_args, **compile_configs.base_input_kwargs)
    graph = infer_node_layer_types(graph)
    # Step 3. Insert communication primitives
    graph = insert_comm_primitives_into_graph(graph, compile_configs.parallelism_plan, forward_pass=True)
    # Step 4. Construct backward graph
    bw_graph = construct_backward_graph(graph)
    bw_graph = insert_comm_primitives_into_graph(bw_graph, compile_configs.parallelism_plan, forward_pass=False)
    # # Step 5. Pipeline stage partition
    # partition_pipeline_stages(graph, compile_configs.parallelism_plan, bw_graph)

    return graph, bw_graph
