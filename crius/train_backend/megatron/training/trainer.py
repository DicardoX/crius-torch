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

        self._init_forward_backward_func()

    def _init_forward_backward_func(self) -> None:
        """ Initialize forward-backward pipeline function. """
        self.forward_backward_func = get_forward_backward_func()
    
    def train(self) -> List[float]:
        """ Training entrypoint. """
        raise NotImplementedError
