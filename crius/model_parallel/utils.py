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

    num_microbatches: int = 1
    """ Number of microbatches in pipeline parallelism. """

    comm_data_type: torch.dtype = torch.float16
    """ Data type used in communication operations. """
    
    tensor_parallel_degree: int = 1
    """ Degree of tensor parallelism. """

    data_parallel_degree: int = 1
    """ Degree of data parallelism. """

    pipeline_stage_partition_method: str = "uniform"
    """ How to partition operators into stages. Options: ['uniform', 'auto', ]. 
    Detailed in `class CriusParallelPlanner`."""

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

    ##############################
    # Runtime Computation Graphs #
    ##############################

    graph: Optional["IRGraph"] = None
    """ Forward IR graph after compilation and pipeline partitioning. """

    bw_graph: Optional["IRGraph"] = None
    """ Backward IR graph after compilation and pipeline partitioning. """
