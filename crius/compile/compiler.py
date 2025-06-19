#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Implementation of Torch-like model compiler. """

from typing import (Any, List, Tuple, Callable)
from dataclasses import dataclass

import torch
from torch import nn
from torch.fx import GraphModule

from crius.compile.graph.transform import (transform_to_ir_graph, insert_comm_primitives_into_graph)
from crius.compile.graph.graph import (IRNode, IRGraph)

__all__ = [
    "CompileConfigs",
    "compile",
]


@dataclass
class CompileConfigs:
    """ Compilation configurations. """
    
    base_input_args: Tuple[Any]
    """ Input arguments of model `.forward()` for IR graph transformation. """

    parallelism_plan: str
    """ Parallelism plan of the model. TODO(chunyu): support inter-stage hybrid parallelism. """

    parallelism_degree: int
    """ Degree of the parallelism plan (e.g., 2 for 2-GPU tensor parallelism). """


def compile(
    model: nn.Module,
    compile_configs: CompileConfigs,
) -> Any:
    """ Python-to-python compilation. 
    
    Args:
        model: Torch-like model to be compiled.
        compile_configs: Compilation configurations.
    """
    # Step 1. Transform model into IR graph
    graph = transform_to_ir_graph(model, *compile_configs.base_input_args)
    
    # Step 2. Insert communication primitives
    graph = insert_comm_primitives_into_graph(
        graph, compile_configs.parallelism_plan, compile_configs.parallelism_degree,
    )

    for node in graph.nodes:
        print(f"Node: {node.name} | Node class: {node.__class__} | OP type: {node.op} | Next: {node.next} | Tar")

    print(graph.owning_module)
    print(graph.owning_module.__class__)
    print(graph.python_code(root_module="self"))

    print(graph._codegen)
