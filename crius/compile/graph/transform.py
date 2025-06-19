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

__all__ = [
    "transform_to_ir_graph",
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


def insert_comm_primitives_into_graph(
    graph: IRGraph,
    parallelism_plan: str,
    parallelism_degree: int,
 ) -> IRGraph:
    """ Insert IR nodes of communication primitives in IR graph, based on the parallelism plan. 
    
    Args:
        graph: IR graph to be inserted with communication primitives.
        parallelism_plan: How the to-be-compiled model would be parallelized.
        parallelism_degree: Degree of the specified parallelism plan.
    """
    # TODO(chunyu): Implement this function.
    return graph
