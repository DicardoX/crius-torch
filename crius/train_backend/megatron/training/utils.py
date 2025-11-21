#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import Callable, Iterable, Optional, List
import random
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import DataLoader
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.optimizer import OptimizerConfig, MegatronOptimizer, get_megatron_optimizer
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig)
from megatron.core.utils import get_attr_wrapped_model

__all__ = [ "setup_model", "get_train_data_iterator", "get_optimizer_and_scheduler", ]


def setup_model(model: MegatronModule, wrap_with_ddp: bool = True) -> List[MegatronModule]:
    """ Prepare model setups. """
    if not isinstance(model, list):
        model = [model]

    megatron_config = get_attr_wrapped_model(model[0], attr="config")
    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    
    # GPU allocation
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # FP16 conversion
    if getattr(megatron_config, "dtype", torch.float16) == torch.float16:
        model = [Float16Module(megatron_config, model_module) for model_module in model]
        # Empty CUDA cache released by FP16 conversion
        torch.cuda.empty_cache()

    if wrap_with_ddp:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=getattr(megatron_config, "grad_reduce_in_fp32", False),
            overlap_grad_reduce=getattr(megatron_config, "overlap_grad_reduce", False),
            use_distributed_optimizer=getattr(megatron_config, "use_distributed_optimizer", False),
            check_for_nan_in_grad=getattr(megatron_config, "check_for_nan_in_grad", False),
        )
        # Wrap model
        model = [
            DistributedDataParallel(
                config=megatron_config,
                ddp_config=ddp_config,
                module=_model_chunk,
                # data_parallel_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                # expert_data_parallel_group=parallel_state.get_data_modulo_expert_parallel_group(),
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(_chunk_idx > 0)
            ) for (_chunk_idx, _model_chunk) in enumerate(model)
        ]
        # Broadcast params from data parallel src rank to other data parallel ranks
        if getattr(megatron_config, "data_parallel_random_init", False):
            for model_chunk in model:
                model_chunk.broadcast_params()

    return model


def get_train_data_iterator(
    vocab_size: int, global_batch_size, num_micro_batches, sequence_length: int, 
    random_seed: int = 123, path_to_cache: str = None, num_iters: int = 10,
) -> Iterable:
    """ Get training data iterator. """
    
    # Dataset
    dc_kwargs = {
        "tokenizer": NullTokenizer(vocab_size),
        "sequence_length": sequence_length,
        "split": None,
        "random_seed": random_seed,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "path_to_cache": path_to_cache,
    }
    dataset_config = GPTDatasetConfig(**dc_kwargs)
    
    def __is_dataset_built_on_rank():
        """ 
        Returns True if the dataset should be built on the current rank and False 
        otherwise. It should be Megatron Core parallelism aware i.e. global rank, 
        local group rank, and virtual rank may inform its return value. 

        When interleaved pipeline is enabled, the dataset should be built on the
        first and last pipeline stage, instead of the virtual pipeline stage.
        """
        return (
            mpu.is_pipeline_first_stage(ignore_virtual=True) or 
            mpu.is_pipeline_last_stage(ignore_virtual=True)
        ) and mpu.get_tensor_model_parallel_rank() == 0

    (train_dataset, _, _) = BlendedMegatronDatasetBuilder(
        cls=MockGPTDataset, sizes=[num_iters * global_batch_size, 0, 0],
        is_built_on_rank=__is_dataset_built_on_rank,
        config=dataset_config,
    ).build()

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=(global_batch_size // num_micro_batches),
        shuffle=True,
    )
    return iter(train_dataloader)


def get_optimizer_and_scheduler(
    model: List[MegatronModule],
    optimizer_str: str = "adam",
    lr: Optional[float] = None,
    min_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    fp16: bool = False,
    bf16: bool = False,
    params_dtype: torch.dtype = torch.float32,
    loss_scale: float = None,
    use_distributed_optimizer: bool = False,
): 
    """ Get optimizer and learning rate scheduler. """
    
    # Optimizer
    oc_kwargs = {
        "optimizer": optimizer_str,
        "lr": lr,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "fp16": fp16,
        "bf16": bf16,
        "params_dtype": params_dtype,
        "loss_scale": loss_scale,
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    opt_config = OptimizerConfig(**oc_kwargs)
    optimizer = get_megatron_optimizer(opt_config, model)

    print(optimizer)
    exit(0)


class NullTokenizer:
    """
    (From megatron.core) Synthetic tokenizer for performance benchmarking and debugging

    Args:
        vocab_size: vocabulary size for embedding
    """

    def __init__(self, vocab_size):
        """ """
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def text_to_ids(self, text):
        """Converts text to ids."""
        return [int(x) for x in text.split(' ')]

    def ids_to_text(self, ids):
        """Converts ids to text."""
        text = [str(x) for x in ids]
        return ' '.join(text)
    
    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Returns offsets."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def unique_identifiers(self) -> OrderedDict:
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        """ """
        raise NotImplementedError

    @property
    def inv_vocab(self):
        """ """
        raise NotImplementedError
    
    @property
    def cls(self):
        """Returns cls token."""
        return -1

    @property
    def sep(self):
        """Returns sep token."""
        return -1

    @property
    def mask(self):
        """Returns mask token."""
        return -1

    @property
    def eod(self):
        """Returns eod token."""
        return self._eod_id
    
    @property
    def additional_special_tokens_ids(self):
        """ """
        return None
