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
logger.setLevel(logging.DEBUG)


@dataclass
class GlobalEnvConfigs:
    """ Global environmental configurations. 
    
    In distributed training, each parallel rank (i.e., each process) maintains its own global environmental 
    configurations. Thus, runtime modified configurations should be boardcasted among all ranks.
    """

    training_backend: str = "megatron"
    """ Training backend used for model training. Options: ['megatron', ] """

    use_shadow_model_loader: bool = False
    """ Whether to use shadow model loader to reduce device memory consumption during model loading."""

    ###########################
    # Resource Configurations #
    ###########################
    num_nodes: int = 1
    """ Number of nodes. """

    num_devices_per_node: int = 1
    """ Number of devices per node. """

    per_device_memory_capacity_in_gb: float = 10.0
    """ Memory capacity (in GB) of the target compute device. """

    peak_fp16_tflops: float = 149.7
    """ Peak FP16 TFLOPS of the target compute device. """

    current_device: torch.device = torch.device("cpu")
    """ Current compute device of this rank, used for tensor instantiation in internal NN modules to replace 
    `torch.cuda.current_device()`. """

    #####################
    # Model Parallelism #
    #####################
    tensor_model_parallel_size: int = 1
    """ Intra-layer model parallelism. Splits tensors across GPU ranks. """

    data_parallel_size: int = 1 
    """ Data parallelism (Torch DDP). Perform gradient sync after backward. """

    pipeline_model_parallel_size: int = 1
    """ Inter-layer model parallelism. Splits transformer layers across GPU ranks. """

    context_parallel_size: int = 1
    """ Context parallelism. """

    remove_p2p_comm_redundancy_in_hybrid_parallel: bool = True
    """ Perform inter-stage P2P communication only between intra-stage's rank 0, rather than
     between ranks of all DP groups. Then, in the destination stage, perform an additional 
     broadcast across all intra-stage ranks. """

    # noninterleave_overlap_p2p_comm: bool = True
    # """ Enable P2P communication overlap in non-interleave (1f1b) pipeline. Useful when P2P bandwidth is limited, 
    # given that native Megatron only supports P2P overlap in interleaved pipeline (as it assumes enough P2P bandwidth). """

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
            "skip_first_n": 0,                              # Skip the first n nodes matched
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

    ########################
    # Input Configurations #
    ########################
    micro_batch_size: int = 0
    """ Micro-batch size. """

    sequence_length: int = 0
    """ Sequence length. """

    ########################
    # Model Configurations #
    ########################
    hidden_size: int = 0
    """ Transformer hidden size. """


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
