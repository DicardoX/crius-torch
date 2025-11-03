#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Global environmental configurations. """

from typing import (Tuple, Dict, Any, List, Callable)
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

import torch

__all__ = [
    "global_configs",
    "logger",
]


# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

#########################
# Global Configurations #
#########################

@dataclass
class GlobalEnvConfigs:
    """ Global environmental configurations. 
    
    In distributed training, each parallel rank (i.e., each process) maintains its own global environmental 
    configurations. Thus, runtime modified configurations should be boardcasted among all ranks.
    """

    training_backend: str = "megatron"
    """ Training backend used for model training. Options: ['megatron', ] """

    ###########################
    # Resource Configurations #
    ###########################
    num_nodes: int = 1
    """ Number of nodes. """

    num_devices_per_node: int = 1
    """ Number of devices per node. """

    #####################
    # Model Parallelism #
    #####################
    tensor_model_parallel_size: int = 1
    """ Intra-layer model parallelism. Splits tensors across GPU ranks. """

    data_parallel_size: int = 1 
    """ Data parallelism (Torch DDP). Perform gradient sync after backward. """

    pipeline_model_parallel_size: int = 1
    """ Inter-layer model parallelism. Splits transformer layers across GPU ranks. """

    ###################
    # Compile Options #
    ###################
    comm_op_insert_rules: Dict[str, Any] = None
    """ Rules for inserting communication operators into the computation graph. 
    The rules are case-specific and should be customized by engineers. 
    
    Format:
    {
        "all_reduce:tp": {
            "after_markers": ['mlp_linear_fc2_weight', ]    # Insert all-reduce after the marker (identify which high-level operator)
            "after_nodes": ['contiguous', ],                # Insert all-reduce after nodes with keywords 'contiguous', which belong to an after_marker
            "interval": 1,                                  # Interval to insert, 1 means inserting every `after_node`
            "pass": 'forward',                              # Which pass is this communication operator inserted ('backward' for DP)
            "parallelism": 'tp',                            # Which parallelism does this communication operator belong to
        },
        ...
    }
    """

    enable_backward_graph_generation: bool = False
    """ Generate backward IR graph with only GEMM and communication operators. 
    
    If false, will direct estimate backward stage latency as twice that of forward one (might be inaccurate).
    """

    only_forward: bool = False
    """ Only conduct forward pass for the model. """

    #################
    # Communication #
    #################
    supported_comm_primitives: List[str] = field(
        default_factory=lambda: ["all_reduce", "all_gather", "reduce_scatter", "all_to_all", ]
    )
    """ Supported communication primitives. """

    comm_op_funcs: Dict[str, Callable] = field(
        default_factory=lambda: {
            "all_reduce": torch.distributed.all_reduce,
        }
    )
    """ Function implementations of communication operators. """


global_configs = GlobalEnvConfigs()
""" Object of global configurations. """


dtype_to_bytes = {
    torch.float32: 4,
    torch.float16: 2,
    torch.float64: 8,
    torch.int32: 4,
    torch.int64: 8,
}
""" Data type to number of bytes mapping. """
