#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Global environmental configurations. """

from typing import (Tuple, Dict, Any, List)
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

    #####################
    # Model Parallelism #
    #####################
    tensor_model_parallel_size: int = 1
    """ Intra-layer model parallelism. Splits tensors across GPU ranks. """

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
        "all_reduce": {
            "after_nodes": ['fc2', ],   # Insert all-reduce after nodes with keywords 'fc2'
            "interval": 1,              # Interval to insert, 1 means inserting every `after_node`
        },
        ...
    }
    """

    #################
    # Communication #
    #################
    supported_comm_primitives: List[str] = field(
        default_factory=lambda: ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]
    )
    """ Supported communication primitives. """


global_configs = GlobalEnvConfigs()
""" Object of global configurations. """
