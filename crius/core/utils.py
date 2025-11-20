#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np
from copy import deepcopy

import torch
from torch import nn

from crius.model_parallel.utils import ParallelismPlan
from crius.compile.graph.graph import IRGraph
from crius.global_envs import global_configs, dtype_to_bytes

__all__ = [
    "estimate_model_memory_access_gb",
]


def estimate_model_memory_access_gb(
    parallelism_plan: ParallelismPlan, graph: IRGraph = None, bw_graph: IRGraph = None,
) -> List[float]:
    """ Estimate memory access of the parallelized model, including weight, activation, 
    optimizer, and gradient. 
    
    Args:
        parallelism_plan: The parallelism plan for the model to be profiled with.
        graph: (Optional) The compiled forward graph (if set, will overwrite that in parallelism_plan).
        bw_graph: (Optional) The compiled backward graph (same as above).
    """
    assert not parallelism_plan.use_gradient_checkpointing, \
        f"Memory estimation with gradient checkpointing is not supported yet."

    stage_weight_mems, stage_act_mems, stage_optim_mems, stage_grad_mems = [
        [0.0 for _ in range(parallelism_plan.pipeline_parallel_degree)] for _ in range(4)
    ]
    if graph is None:
        graph, bw_graph = parallelism_plan.graph, parallelism_plan.bw_graph

    for node in graph.nodes:
        # if "tensor_meta" not in node.meta:
        #     continue    # Skip non-tensor node
        # Since we deepcopy precompiled graph yet cannot deepcopy tensor_meta, we directly 
        # use node.input_shapes/output_shapes to estimate weight shape
        weight_shape = None
        if node.layer_type == "gemm":
            # Weight
            weight_shape = [node.input_shapes[0][-1], node.output_shapes[0][-1]]
            if node.stage_index < len(stage_weight_mems):
                stage_weight_mems[node.stage_index] += float(
                    (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                    parallelism_plan.tensor_parallel_degree
                ) / (1024 ** 3)   # GB

            # Optimizer states
            # Adam optimizer by default, with 2 states per weight parameter
            if node.stage_index < len(stage_optim_mems):
                stage_optim_mems[node.stage_index] += float(
                    2 * (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                    parallelism_plan.tensor_parallel_degree
                ) / (1024 ** 3)   # GB

            # Gradient
            if node.stage_index < len(stage_grad_mems):
                stage_grad_mems[node.stage_index] += float(
                    (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                    parallelism_plan.tensor_parallel_degree
                ) / (1024 ** 3)   # GB
        
        # Activation
        # FIXME(chunyu): Incomplete activation memory estimation, should consider more layer types.
        if node.layer_type in ("gemm", "bmm", "baddbmm", "layernorm", ):
            for out_shape in node.output_shapes:
                if node.stage_index < len(stage_act_mems):
                    stage_act_mems[node.stage_index] += float(
                        (np.prod(list(out_shape)) * dtype_to_bytes.get(parallelism_plan.comm_data_type, 4)) / 
                        parallelism_plan.tensor_parallel_degree
                    ) / (1024 ** 3)   # GB

    max_stage_weight_mem, max_stage_act_mem, max_stage_optim_mem, max_stage_grad_mem = max(stage_weight_mems), \
        max(stage_act_mems), max(stage_optim_mems), max(stage_grad_mems)
    return max_stage_weight_mem, max_stage_act_mem, max_stage_optim_mem, max_stage_grad_mem
