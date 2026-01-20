#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Utilities for model parallelism. """

from typing import Dict, List, Optional
from dataclasses import dataclass

import torch

__all__ = [
    "ParallelismPlan",
]


@dataclass
class ParallelismPlan:
    """ Parallelism plan for hybrid model parallelism. """

    pipeline_type: str = "1f1b"
    """ Type of pipeline parallelism. """

    pipeline_parallel_degree: int = 1
    """ Degree of pipeline parallelism. """

    num_layers: int = 1
    """ Number of transformer layers in the LLM model. """

    num_attn_heads: int = 1
    """ Number of attention heads in the LLM model. """

    num_microbatches: int = 1
    """ Number of microbatches in pipeline parallelism. """

    comm_data_type: torch.dtype = torch.float16
    """ Data type used in communication operations. """
    
    tensor_parallel_degree: int = 1
    """ Degree of tensor parallelism. """

    data_parallel_degree: int = 1
    """ Degree of data parallelism. """

    use_pytorch_ddp_data_partitioning: bool = True
    """ Use PyTorch's DDP style batch data partitioning, i.e., each DP replica executes a full microbatch, 
    respectively, instead of internally sharding the same microbatch into #DP-degree partitions. For example,
    assuming 16 microbatches and DP=2, each DP replica executes 8 microbatches and the total number of microbatches
    for the per-DP pipeline becomes 8. """

    use_global_grad_sync_dp_backward: bool = True
    """ When DP degree < #GPUs (i.e., in hybrid parallelism), performing gradient synchronization across all GPUs, 
    rather than only across DP replicas. """

    forced_grad_sync_latency_scale_factor: float = 1.0
    """ Scale factor for the estimated gradient synchronization latency across all GPUs during backward pass.
    Used (and to be tuned) when in multi-node setting, because we find that the cross-node grad sync overhead of DDP
    is quite larger than intra-node one (not existed in intra-node DDP). """

    forced_p2p_latency_scale_factor: float = 1.0
    """ Scale factor for the estimated P2P communication latency during pipeline execution. """

    pipeline_stage_partition_method: str = "uniform"
    """ How to partition operators into stages. Options: ['uniform', 'auto', ]. 
    Detailed in `class CriusParallelPlanner`."""

    pipeline_num_layers_per_stage: List[int] = None
    """ Number of transformer layers in each stage. Pre-loop (e.g., embedding) and post-loop (e.g., output projection) 
    operators are considered as the first and last layer (The length of this list = num_layers + 2). """

    pipeline_overlap_p2p_comm: bool = False
    """ Whether to overlap P2P communication with computation in pipeline parallelism. """

    pipeline_forced_p2p_comm_overhead_ms: float = -1.0
    """ Forcibly set the p2p communication overhead, used when p2p overhead is non-negligible and 
    unpredictable at runtime (could check it via torch profiler). """

    pipeline_use_boundary_markers: bool = False
    """ Whether to use boundary markers to indicate the start and end of a layer in the IR graph. """

    pipeline_boundary_start_marker: List[str] = None
    """ List of node targets that indicate the start of a layer (e.g., ['getattr', 'input_layernorm', ]). """

    pipeline_boundary_end_marker: List[str] = None
    """ List of node targets that indicate the end of a layer (e.g., ['dropout', 'add', ]). """

    pipeline_num_preprocess_nodes: int = -1
    """ Number of preprocess nodes (e.g., arg0) before the loop of layers (e.g., transformer layers). """

    pipeline_num_postprocess_nodes: int = -1
    """ Number of postprocess nodes (e.g., output) after the loop of layers (e.g., transformer layers). """

    op_to_layer_mapping: Dict[str, int] = None
    """ (Deprecated) Mapping from operator name to layer index.
    `{"fc1": 0, "fc2": 1}` means `fc1` is in layer 0 and `fc2` is in layer 1"""

    layer_to_stage_mapping: List[List[int]] = None
    """ (Deprecated) Mapping from layer index to stage index. 
    `[[0, 1], [2, 3]]` means layer 0/1 are in stage 0 and layer 2/3 are in stage 1. """

    ########################
    # Optimization Options #
    ########################
    use_gradient_checkpointing: bool = False
    """ Whether to enable gradient checkpointing (i.e., recomputation) for memory reduction. """

    ##############################
    # Runtime Computation Graphs #
    ##############################
    graph: Optional["IRGraph"] = None
    """ Forward IR graph after compilation and pipeline partitioning. """

    bw_graph: Optional["IRGraph"] = None
    """ Backward IR graph after compilation and pipeline partitioning. """

    def __post_init__(self):
        if self.use_pytorch_ddp_data_partitioning:
            # Each DP replica executes a full microbatch
            assert self.num_microbatches % self.data_parallel_degree == 0, \
                f"Number of microbatches ({self.num_microbatches}) must be divisible by " + \
                f"DP degree ({self.data_parallel_degree})."        
            self.num_microbatches = self.num_microbatches // self.data_parallel_degree
