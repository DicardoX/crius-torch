#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Implementation of Torch-like model compiler. """

from typing import (Any, List, Tuple, Callable)
from dataclasses import dataclass

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
    "compile",
]


@dataclass
class CompileConfigs:
    """ Compilation configurations. """
    
    base_input_args: Tuple[Any]
    """ Input arguments of model `.forward()` for IR graph transformation. """

    parallelism_plan: ParallelismPlan
    """ Parallelism plan of the model. """


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
    graph = transform_to_ir_graph(model, *compile_configs.base_input_args)
    # Step 2. Propagate per-node input/output shapes
    graph = propagate_node_shapes(graph, *compile_configs.base_input_args)
    graph = infer_node_layer_types(graph)
    # Step 3. Insert communication primitives
    graph = insert_comm_primitives_into_graph(graph, compile_configs.parallelism_plan, forward_pass=True)
    # Step 4. Construct backward graph
    bw_graph = construct_backward_graph(graph)
    bw_graph = insert_comm_primitives_into_graph(bw_graph, compile_configs.parallelism_plan, forward_pass=False)

    exit(0)

    return graph, bw_graph
