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

from crius.global_envs import logger
from crius.train_backend.megatron.training.utils import log_on_rank, Timer

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
        self.timer = Timer()

        self._init_forward_backward_func()

    def _init_forward_backward_func(self) -> None:
        """ Initialize forward-backward pipeline function. """
        self.forward_backward_func = get_forward_backward_func()
    
    def train_step(self) -> Tuple[List[torch.Tensor]]:
        """ Training step for one iteration of a mini-batch. """

        # This model partition belongs to the last stage of the pipeline
        last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        # Zero grad of model and optimizer
        if self.optimizer:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()  # Defined in `DistributedDataParallel`
            self.optimizer.zero_grad()
        
        self.timer.start("train_step", use_cuda_event=True)

        # Forward-backward pass
        assert self.forward_backward_func is not None, \
            "Forward-backward function of the trainer is not set."
        tr_loss = self.forward_backward_func(
            forward_step_func=self.forward_step_func,
            data_iterator=self.train_data_iterator,
            model=self.model,
            num_microbatches=self.megatron_config.num_micro_batches,
            seq_length=self.megatron_config.sequence_length,
            micro_batch_size=(
                self.megatron_config.global_batch_size // 
                self.megatron_config.num_micro_batches
            ),
            forward_only=self.forward_only,
        )

        # FIXME(chunyu): Currently we only consider forward-backward computation without
        # other processes such as optimizer step.
        self.timer.stop("train_step", use_cuda_event=True)

        # Unscale grad
        self._maybe_unscale_grad()

        if self.forward_only:
            return (tr_loss, ) if last_stage else (None, )
        
        # Update parameters
        update_successful = False
        if isinstance(self.optimizer, MegatronOptimizer):
            (update_successful, grad_norm, num_zeros_in_grad) = self.optimizer.step()
        else:
            raise TypeError(f"Invalid optimizer type: {self.optimizer.__class__}")

        # Update learning rate
        if update_successful:
            increment = (
                self.megatron_config.global_batch_size * 
                parallel_state.get_data_parallel_world_size()
            )
            self.lr_scheduler.step(increment=increment)
        
        torch.cuda.synchronize()
        return (tr_loss, ) if last_stage else (None, )

    def train(self, report_metrics: bool = True) -> List[float]:
        """ Training entrypoint. """
        
        # Turn on training mode which enables dropout
        for model_chunk in self.model:
            model_chunk.train()

        # Setup training configurations
        megatron_config = get_attr_wrapped_model(self.model[0], attr="config")
        # - Gradient scale function
        megatron_config.grad_scale_func = self.optimizer.scale_loss if self.optimizer else None
        # - Finalize model gradients (i.e., grad sync)
        megatron_config.finalize_model_grads_func = finalize_model_grads if self.optimizer else None

        # Training loop
        tr_losses = []
        for iter_index in range(self.megatron_config.num_iters):
            (tr_loss, ) = self.train_step()
            tr_losses.append(tr_loss)

            log_on_rank(
                msg=f"Losses of the training iteration ({iter_index}): {tr_loss}",
                rank=[0, parallel_state.get_pipeline_model_parallel_last_rank()],
                group_type=["tensor", "pipeline"],
                logger=logger,
                log_level="info",
            )

        if report_metrics:
            raw_record = self.timer._records.pop("train_step")
            e2e_iter_times = [round(_r.elapsed, 3) for _r in raw_record]
            e2e_iter_times = e2e_iter_times[1:]     # Warmup iter excluded
            log_on_rank(
                msg=f"Average end-to-end iteration time (ms): {np.mean(e2e_iter_times)}",
                rank=[0, 0],
                group_type=["tensor", "pipeline"],
                logger=logger,
                log_level="info",
            )

        return tr_losses

    def _maybe_unscale_grad(self) -> None:
        """ Unscale gradient if gradient scaler is enabled. """

        if (not hasattr(self.optimizer, "grad_scaler") or self.optimizer.grad_scaler is None):
            return

        main_grads = []
        for model_chunk in self.model:
            for param in model_chunk.parameters():
                if param.grad is not None:
                    main_grads.append(param.grad.data)

        found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, found_inf, self.optimizer.grad_scaler.inv_scale,
        )
