#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Global environmental configurations. """

from typing import (Tuple)
from dataclasses import dataclass
from collections import OrderedDict

import torch

__all__ = [
    "global_configs",
]


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


global_configs = GlobalEnvConfigs()
""" Object of global configurations. """
