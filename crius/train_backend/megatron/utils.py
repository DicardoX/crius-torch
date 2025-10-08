#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Utilities for Megatron backend. """

import os
import torch
from megatron.core import parallel_state

from crius.global_envs import logger

__all__ = [
    "initialize_distributed",
    "WrappedTensor",
]


def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    """ Initialize the Megatron backend. 
    
    This function is used to initialize the Megatron backend for distributed training.
    """

    torchrun_world_size = int(os.environ['WORLD_SIZE'])
    num_available_gpus = torch.cuda.device_count()
    if torchrun_world_size != num_available_gpus:
        logger.warning(f"=========> The number of GPUs available ({num_available_gpus}) does not match the WORLD_SIZE ({torchrun_world_size}).")

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=torchrun_world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)    
    logger.info(f"=========> Megatron distributed training initialized with rank {rank} and world size {torchrun_world_size}.")


class WrappedTensor:
    """
    A wrapper for tensors that enables caller functions to pass an indirect reference
    to callee functions. By wrapping the tensor, the caller's direct reference is removed,
    allowing the tensor to be garbage collected once the callee unwraps and frees it.
    """

    def __init__(self, tensor: torch.Tensor):
        self._wrapper = [tensor]

    def unwrap(self):
        """
        Returns the wrapped tensor while deleting the internal reference.
        Can only be called once.
        """
        if len(self._wrapper) == 0:
            raise RuntimeError(f"WrappedTensor has already been unwrapped")
        return self._wrapper.pop(0)
