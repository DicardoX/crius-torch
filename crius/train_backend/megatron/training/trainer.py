#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Trainer implementation for Megatron. """

import os
from typing import (
    TYPE_CHECKING, Callable, Dict, Generator, List, Any,  
    Optional, Tuple, Type, Union, cast, Iterator)
import enum
from contextlib import nullcontext
import logging
from collections import OrderedDict
import time
import numpy as np

import torch
import torch.distributed

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import (
    get_forward_backward_func, set_current_microbatch)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig)
from megatron.core.utils import get_attr_wrapped_model

from crius.global_envs import global_configs

__all__ = [ "ModelType", "Trainer", ]


class ModelType(enum.Enum):
    """ Type class of the model type. """

    GPT = enum.auto()
    LLAMA = enum.auto()


class Trainer:
    """
    Trainer class that wraps Megatron's training procedure.

    Args:
        trainer_configs: The transformer config object in Megatron, with monkey patched attributes.
        forward_step_func: The forward step function for the model that takes the
            data iterator as the first argument, and model as the second. Detailed in 
            the docstring of `megatron.core.pipeline_parallel.schedules.forward_step()`.
        model: The model object defined in the format of MegatronModule.
        model_type: Type of the model to be trained.
        train_data_iterator: The iterator of training data.
        optimizer: The optimizer for model training.
        opt_param_scheduler: The learning rate scheduler.
        pipeline_strategy: Strategy of the pipeline execution.
        forward_only: Only perform forward pass.
    """

    def __init__(
        self,
        megatron_transformer_config: TransformerConfig,
        forward_step_func: Callable,
        model: Union[MegatronModule, List[MegatronModule]],
        model_type: ModelType,
        train_data_iterator: Iterator,
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
        pipeline_strategy: str = None,
        forward_only: bool = False,
    ) -> None:
        """ Initialize a Trainer object. """

        self.megatron_config = megatron_transformer_config
        self.model = model
        self.model_type = model_type

        self.train_data_iterator = train_data_iterator
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.pipeline_strategy = pipeline_strategy
        self.forward_only = forward_only

        self.forward_step_func = forward_step_func
        self.forward_backward_func = None

        self._setup_model()
        self._init_forward_backward_func()

    def _setup_model(self, wrap_with_ddp=True):
        """ Prepare model setups. """
        if not isinstance(self.model, list):
            self.model = [self.model]

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in self.model:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        
        # GPU allocation
        for model_module in self.model:
            model_module.cuda(torch.cuda.current_device())

        # FP16 conversion
        if getattr(self.megatron_config, "dtype", torch.float16) == torch.float16:
            self.model = [Float16Module(self.megatron_config, model_module) for model_module in self.model]
            # Empty CUDA cache released by FP16 conversion
            torch.cuda.empty_cache()

        if wrap_with_ddp:
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=getattr(self.megatron_config, "grad_reduce_in_fp32", False),
                overlap_grad_reduce=getattr(self.megatron_config, "overlap_grad_reduce", False),
                use_distributed_optimizer=getattr(self.megatron_config, "use_distributed_optimizer", False),
                check_for_nan_in_grad=getattr(self.megatron_config, "check_for_nan_in_grad", False),
            )
            # Wrap model
            self.model = [
                DistributedDataParallel(
                    config=self.megatron_config,
                    ddp_config=ddp_config,
                    module=_model_chunk,
                    # data_parallel_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    # expert_data_parallel_group=parallel_state.get_data_modulo_expert_parallel_group(),
                    # Turn off bucketing for model_chunk 2 onwards, since communication for these
                    # model chunks is overlapped with compute anyway.
                    disable_bucketing=(_chunk_idx > 0)
                ) for (_chunk_idx, _model_chunk) in enumerate(self.model)
            ]
            # Broadcast params from data parallel src rank to other data parallel ranks
            if getattr(self.megatron_config, "data_parallel_random_init", False):
                for model_chunk in self.model:
                    model_chunk.broadcast_params()

    def _init_forward_backward_func(self) -> None:
        """ Initialize forward-backward pipeline function. """
        self.forward_backward_func = get_forward_backward_func()
    
    def train(self) -> List[float]:
        """ Training entrypoint. """
        raise NotImplementedError
