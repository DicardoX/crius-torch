#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import torch
from megatron.core import parallel_state

from crius.global_envs import logger

__all__ = [
    "initialize_distributed",
]


def initialize():
    """ Initialize the Megatron backend. 
    
    This function is used to initialize the Megatron backend for distributed training.
    """
    


def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    """ Initialize the Megatron backend. 
    
    This function is used to initialize the Megatron backend for distributed training.
    """
    
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)    
    logger.info(f"=========> Megatron distributed training initialized with rank {rank} and world size {world_size}.")
