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
    "estimate_per_op_fwd_flops",
    "estimate_per_op_fwd_memory_access_gb",
    "estimate_model_memory_access_gb",
]


def estimate_per_op_fwd_memory_access_gb(proto_plan: ParallelismPlan, graph: IRGraph) -> None:
    """ Estimate per-operator memory access of the model, including weight and activation as they are 
    involved in forward pass. """

    for node in graph.nodes:
        node.memory_access = 0.0
        # Weight
        weight_shape = None
        if node.layer_type == "gemm":
            weight_shape = [node.input_shapes[0][-1], node.output_shapes[0][-1]]
            node.memory_access += float(
                (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4))
            ) / (1024 ** 3)   # GB
        
        # Activation
        # FIXME(chunyu): Incomplete activation memory estimation, should consider more layer types.
        if node.layer_type in ("gemm", "bmm", "baddbmm", "layernorm", ):
            for out_shape in node.output_shapes:
                node.memory_access += float(
                    (np.prod(list(out_shape)) * dtype_to_bytes.get(proto_plan.comm_data_type, 4))
                ) / (1024 ** 3)   # GB


def estimate_per_op_fwd_flops(graph: IRGraph) -> None:
    """ Estimate per-operator FLOPs of the model in forward pass. """
    for node in graph.nodes:
        node.flops = 0.0

        if node.layer_type == "gemm":
            # Shape: [s, b, h1] x [h1, h2]
            (s, b, h1) = node.input_shapes[0]
            h2 = node.input_shapes[1][-1]
            node.flops = (s * b * h2) * (2 * h1 - 1)
            
        elif node.layer_type == "bmm":
            # Shape: [b, s, s] x [b, s, h1] 
            (b, s, h1) = node.input_shapes[1]
            node.flops = 2 * b * s * s * h1

        elif node.layer_type == "baddbmm":
            # Shape: [b, s, s] + [b, s, h1] x [b, h1, s]
            (b, s, h1) = node.input_shapes[1]
            node.flops = 2 * b * s * s * h1 + b * s * s


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
            stage_weight_mems[node.stage_index] += float(
                (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                parallelism_plan.tensor_parallel_degree
            ) / (1024 ** 3)   # GB

            # Optimizer states
            # Adam optimizer by default, with 2 states per weight parameter
            stage_optim_mems[node.stage_index] += float(
                2 * (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                parallelism_plan.tensor_parallel_degree
            ) / (1024 ** 3)   # GB

            # Gradient
            stage_grad_mems[node.stage_index] += float(
                (np.prod(weight_shape) * dtype_to_bytes.get(node.dtype, 4)) / 
                parallelism_plan.tensor_parallel_degree
            ) / (1024 ** 3)   # GB
        
        # Activation
        # FIXME(chunyu): Incomplete activation memory estimation, should consider more layer types.
        if node.layer_type in ("gemm", "bmm", "baddbmm", "layernorm", ):
            for out_shape in node.output_shapes:
                stage_act_mems[node.stage_index] += float(
                    (np.prod(list(out_shape)) * dtype_to_bytes.get(parallelism_plan.comm_data_type, 4)) / 
                    parallelism_plan.tensor_parallel_degree
                ) / (1024 ** 3)   # GB

    max_stage_weight_mem, max_stage_act_mem, max_stage_optim_mem, max_stage_grad_mem = max(stage_weight_mems), \
        max(stage_act_mems), max(stage_optim_mems), max(stage_grad_mems)
    return max_stage_weight_mem, max_stage_act_mem, max_stage_optim_mem, max_stage_grad_mem
