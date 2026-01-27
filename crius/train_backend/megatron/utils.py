#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Utilities for Megatron backend. """

import os
from typing import Optional, Callable
import operator
from contextlib import contextmanager, nullcontext
from functools import lru_cache, reduce, wraps

import torch
from megatron.core import parallel_state
# from megatron.core.parallel_state import _GLOBAL_MEMORY_BUFFER
from megatron.core.utils import GlobalMemoryBuffer

from crius.global_envs import logger, global_configs

__all__ = [
    "initialize_virtual_distributed",
    "initialize_distributed",
    "WrappedTensor",
]


def initialize_virtual_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1,
    data_parallel_size: int = 1, num_nodes: int = 1, num_devices_per_node: int = 1,
    init_cpu_process_group: bool = True, skip_distributed_init: bool = True,
):
    """ Initialize a virtual Megatron backend without ALL underlying physical deivces. 

    By default, this function initializes a multi-process CPU process group. This function 
    should be called by `torchrun --nproc-per-node #proc [python script]`.
    
    This function is used to initialize the Megatron backend for device-free operations,
    such as device-free profiling or tuning. Notably, real training cannot be performed 
    with this virtual backend. 

    Args:
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        data_parallel_size: Degree of data parallelism.
        num_nodes: Number of virtual nodes allocated for the training task.
        num_devices_per_node: Number of virtual devices per node.
        init_cpu_process_group: Whether to initialize a multi-process CPU process group.
        skip_distributed_init: Whether to skip the distributed initialization. Used when not enough
            nodes are allocated for the virtual backend (e.g., only have 1 node while profiling multi-node).
    """

    if skip_distributed_init:
        logger.info(f"=========> Skipping distributed initialization for virtual Megatron backend.")
        torch.distributed.init_process_group(world_size=1, backend='gloo')
        parallel_state.initialize_model_parallel()
        return

    assert init_cpu_process_group
    torchrun_world_size = int(os.environ['WORLD_SIZE'])
    assert torchrun_world_size == max(
        tensor_model_parallel_size * pipeline_model_parallel_size * data_parallel_size, 
        num_nodes * num_devices_per_node,
    ), f"The world size of torchrun ({torchrun_world_size}) is not properly set."
    rank = int(os.environ['RANK'])
    torch.distributed.init_process_group(world_size=torchrun_world_size, rank=rank, backend='gloo')

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel()
    # Overwrite Megatron's global memory buffer implementation as it forces CUDA tensors when adding a new 
    # tensor to the buffer.
    assert isinstance(parallel_state.get_global_memory_buffer(), GlobalMemoryBuffer)
    parallel_state._GLOBAL_MEMORY_BUFFER = CPUCompatibleGlobalMemoryBuffer()
    assert isinstance(parallel_state.get_global_memory_buffer(), CPUCompatibleGlobalMemoryBuffer)
    logger.info(f"=========> Megatron distributed training initialized with rank {rank} and world size {torchrun_world_size}.")


def initialize_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, 
    virtual_pipeline_model_parallel_size: int = 1, tp_comm_overlap: bool = False,
):
    """ Initialize the Megatron backend. 
    
    This function is used to initialize the Megatron backend for distributed training.
    """

    torchrun_world_size = int(os.environ['WORLD_SIZE'])
    num_available_gpus = torch.cuda.device_count()
    # if torchrun_world_size != num_available_gpus:
    #     logger.warning(f"=========> The number of GPUs available ({num_available_gpus}) does not match the WORLD_SIZE ({torchrun_world_size}).")

    # Torch setup for distributed training
    local_rank = int(os.environ['LOCAL_RANK'])
    device_id = torch.device(f'cuda:{local_rank}')
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(world_size=torchrun_world_size, rank=global_rank, backend="nccl")

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size,
    )    
    logger.info(f"=========> Megatron distributed training initialized with rank {global_rank} and world size {torchrun_world_size}.")

    if tp_comm_overlap:
        # Overlap TP communication
        _initialize_tp_communicators()


def _initialize_tp_communicators():
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
            "'transformer_engine' packages"
        )

    ub_cfgs = {}
    input_shape = [
        (global_configs.sequence_length * global_configs.micro_batch_size) 
            // global_configs.context_parallel_size,
        global_configs.hidden_size,
    ]

    # We create a MPI process group, which is needed to bootstrap the pipelined
    # tensor-model-parallel communication overlap.
    # We got: RuntimeError: Distributed package doesn't have MPI built in. MPI is only 
    #         included if you build PyTorch from source on a host that has MPI installed.
    # Thus we turn to use NCCL as the bootstrap backend.
    torch.distributed.new_group(backend='nccl')
    te_module.base.initialize_ub(
        shape=input_shape, tp_size=global_configs.tensor_model_parallel_size,
        use_fp8=False, ub_cfgs=ub_cfgs, bootstrap_backend="nccl",
    )


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


class CPUCompatibleGlobalMemoryBuffer:
    """
    A CPU-compatible version of Megatron's GlobalMemoryBuffer to avoid dynamic memory allocations.
    
    Caller should ensure that buffers of the same name are not used concurrently.
    """

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context: Optional[Callable] = None):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    # device=torch.cuda.current_device(),
                    device=global_configs.current_device,
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
