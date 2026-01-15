#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Callable, Iterator, Optional, List, Tuple, Union, Dict
import random
from collections import OrderedDict
import numpy as np
import time
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import DataLoader
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.optimizer import OptimizerConfig, MegatronOptimizer, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig)
from megatron.core.utils import get_attr_wrapped_model

__all__ = [
    "setup_model", 
    "get_train_data_iterator", 
    "get_optimizer_and_scheduler", 
    "get_input_batch",
    "log_on_rank",
    "Timer",
]

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


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
) -> Iterator:
    """ Get training data iterator. """
    
    # Dataset
    dc_kwargs = {
        "tokenizer": build_tokenizer(),
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
            parallel_state.is_pipeline_first_stage(ignore_virtual=True) or 
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ) and parallel_state.get_tensor_model_parallel_rank() == 0
    
    print(f"\n\nRanks: {parallel_state.get_all_ranks()} building dataset: {__is_dataset_built_on_rank()}\n\n")

    (train_dataset, _, _) = BlendedMegatronDatasetBuilder(
        cls=MockGPTDataset, sizes=[num_iters * global_batch_size, 0, 0],
        is_built_on_rank=__is_dataset_built_on_rank,
        config=dataset_config,
    ).build()

    exit(0)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=(global_batch_size // num_micro_batches),
        shuffle=False,
    )

    exit(0)

    return iter(train_dataloader)


def build_tokenizer(
    tokenizer_type: str = "SentencePieceTokenizer",
    tokenizer_model_file_path: str = f"{CUR_PATH}/tokenizer/tokenizer.model",
):
    """ Build the tokenizer for transformer model. """
    if tokenizer_type == "SentencePieceTokenizer":
        from crius.train_backend.megatron.training.megatron_tokenizer import _GPTSentencePieceTokenizer
        tokenizer = _GPTSentencePieceTokenizer(tokenizer_model_file_path)
    else:
        raise TypeError(f"Invalid tokenizer type: {tokenizer_type}")

    return tokenizer


def get_optimizer_and_scheduler(
    model: List[MegatronModule],
    global_batch_size: int,
    num_iters: int,
    optimizer_str: str = "adam",
    lr: Optional[float] = 1e-3,
    max_lr: Optional[float] = 1e-3,
    min_lr: Optional[float] = 1e-4,
    weight_decay: float = 0.01,
    fp16: bool = False,
    bf16: bool = False,
    params_dtype: torch.dtype = torch.float32,
    loss_scale: float = None,
    use_distributed_optimizer: bool = False,
    num_lr_warmup_iters: int = 0,
    lr_decay_style: str = "cosine",
) -> Tuple[MegatronOptimizer, OptimizerParamScheduler]:
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

    # Learning rate scheduler
    lr_warmup_steps = num_lr_warmup_iters * global_batch_size
    lr_decay_steps = weight_decay_steps = num_iters * global_batch_size
    opt_param_scheduler = OptimizerParamScheduler(
        optimizer=optimizer, init_lr=lr, max_lr=max_lr,
        min_lr=min_lr, lr_warmup_steps=lr_warmup_steps, 
        lr_decay_steps=lr_decay_steps, lr_decay_style=lr_decay_style,
        start_wd=0, end_wd=0,
        wd_incr_steps=weight_decay_steps, wd_incr_style="constant",
    )

    return optimizer, opt_param_scheduler


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
        return [int(x) % self._vocab_size_without_eod for x in text.split(' ')]

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


def get_input_batch(
    data_iterator: Iterator,
    micro_batch_size: int,
    seq_len: int,
    pipeline_model_parallel_size: int = 1,
    create_attention_mask_in_dataloader: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """ Get one batch of tokens, labels, loss_mask, attention_mask and position_ids. """

    if (
        (not parallel_state.is_pipeline_first_stage()) and
        (not parallel_state.is_pipeline_last_stage())
    ):
        # Intermediate stages with intermediate generated tensors as input
        return (None, None, None, None, None)
    
    # Get batch based on the rank of tensor parallel of this model partition
    batch = get_batch_on_this_tp_rank(
        data_iterator, micro_batch_size, seq_len, pipeline_model_parallel_size, 
        create_attention_mask_in_dataloader,
    )

    return (
        batch["tokens"], batch["labels"], batch["loss_mask"], batch["attention_mask"], batch["position_ids"],
    )


def get_batch_on_this_tp_rank(
    data_iterator: Iterator,
    micro_batch_size: int,
    seq_len: int,
    pipeline_model_parallel_size: int = 1,
    create_attention_mask_in_dataloader: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """ Get the input batch of this tensor parallel rank. """

    def __broadcast(item):
        """ Broadcast item across tensor parallel group. """
        if item is not None:
            torch.distributed.broadcast(
                tensor=item, 
                src=parallel_state.get_tensor_model_parallel_src_rank(), 
                group=parallel_state.get_tensor_model_parallel_group(),             
            )

    if parallel_state.get_tensor_model_parallel_rank() == 0:
        # Rank 0 of all tensor parallel ranks
        # Data item
        data = next(data_iterator) if data_iterator is not None else None

        # Construct batch
        batch = {
            "tokens": data["tokens"].cuda(non_blocking = True),
            "labels": data["labels"].cuda(non_blocking = True),
            "loss_mask": data["loss_mask"].cuda(non_blocking = True),
            "attention_mask": None if "attention_mask" not in data \
                else data["attention_mask"].cuda(non_blocking = True),
            "position_ids": data["position_ids"].cuda(non_blocking = True)
        }

        # Broadcast data to other ranks
        if pipeline_model_parallel_size == 1:
            # No pipeline
            __broadcast(batch["tokens"])
            __broadcast(batch["labels"])
            __broadcast(batch["loss_mask"])
            __broadcast(batch["attention_mask"])
            __broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_first_stage():
            # First stage with pipeline
            __broadcast(batch["tokens"])
            __broadcast(batch["attention_mask"])
            __broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_last_stage():
            # Last stage with pipeline
            __broadcast(batch["labels"])
            __broadcast(batch["loss_mask"])
            __broadcast(batch["attention_mask"])

    else:
        # Other ranks within tensor parallel
        # Placeholders
        tokens = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.float32, device=torch.cuda.current_device(),
        )
        position_ids = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        attention_mask=torch.empty(
            (micro_batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=torch.cuda.current_device(),
        ) if create_attention_mask_in_dataloader else None

        # Receive data from rank 0
        if pipeline_model_parallel_size == 1:
            # No pipeline
            __broadcast(tokens)
            __broadcast(labels)
            __broadcast(loss_mask)
            __broadcast(attention_mask)
            __broadcast(position_ids)

        elif parallel_state.is_pipeline_first_stage():
            # First stage with pipeline
            labels=None
            loss_mask=None
            __broadcast(tokens)
            __broadcast(attention_mask)
            __broadcast(position_ids)

        elif parallel_state.is_pipeline_last_stage():
            # Last stage with pipeline
            tokens=None
            position_ids=None
            __broadcast(labels)
            __broadcast(loss_mask)
            __broadcast(attention_mask)

        # Construct batch
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    return batch


def log_on_rank(
    msg: str, 
    rank: Union[int, List[int]], 
    group_type: Union[str, List[str]], 
    logger: logging.Logger,
    log_level: str = "info",
): 
    """ Log message on the target rank. """
    
    if isinstance(rank, int):
        rank = [rank]
    if isinstance(group_type, str):
        group_type = [group_type]
    
    if log_level == "info":
        log_func = logger.info
    
    if torch.distributed.is_initialized():
        # Get rank based on parallel group type
        this_ranks = []
        for t in group_type:
            if t == "tensor":
                this_ranks.append(parallel_state.get_tensor_model_parallel_rank())
            elif t == "pipeline":
                this_ranks.append(parallel_state.get_pipeline_model_parallel_rank())
            else:
                raise TypeError(f"Invalid parallel group type: {t}")

        if all([this_ranks[_i] == rank[_i] for _i in range(len(rank))]):
            log_func(msg)
    else:
        log_func(msg)


#####################
#   Timer Methods   #
#####################

@dataclass
class TimeRecord:
    """ Record of one timing operation. """

    opcode: str = None
    """ Opcode to identify the operation to be timed. """

    start_t: float = None
    """ Start timestamp obtained from `timer.time()`. """

    stop_t: float = None
    """ Stop timestamp obtained from `timer.time()`. """

    elapsed: float = None
    """ Elapsed time of the operation. """

    start_event: torch.cuda.Event = None
    """ CUDA event to start the timing. """

    stop_event: torch.cuda.Event = None
    """ CUDA event to stop the timing. """

    def __post_init__(self):
        """ Post initialization. """
        assert self.opcode, "The opcode of this time record must be set for identification."


class Timer:
    """ Class of global timer of each process. """

    def __init__(self) -> None:
        """ Initialize a timer object. """

        self._records: Dict[str, List[TimeRecord]] = {}
        self._barrier_group = None

    def set_barrier_group(self, barrier_group: torch.distributed.ProcessGroup):
        """ Sets the barrier group used to synchronize ranks. """
        self._barrier_group = barrier_group
    
    def _maybe_sync(self, barrier: bool = False):
        """ Synchronize ranks and local CUDA events. """

        if barrier:
            # Synchronize all ranks
            torch.distributed.barrier(group=self._barrier_group)
        # Synchronize local cuda events
        torch.cuda.synchronize()

    def start(self, opcode: str, use_cuda_event: bool = False, barrier: bool = False):
        """ Start a timer operation. 

        This can introduce additional overhead as some synchronization and barrier operations 
        are called to measure time cost precisely. 
        
        Args:
            opcode: The opcode as the identification of the operation to be timed.
            use_cuda_event: Use `torch.cuda.Event()` to record the elapsed time of a GPU computation.
                             Compared to `barrier` option, this can measure the time cost of a GPU computation
                             more precisely.
            barrier: Use `torch.distributed.barrier()` to synchronize all ranks in the barrier group.
                      When pipeline is used, it should be carefully used (by setting barrier group) to
                      avoid deadlock between blocking of the communication primitives and the barrier
                      option in the timer.
        """

        assert (
            opcode not in self._records or 
            sum([int(_r.elapsed is None) for _r in self._records[opcode]]) == 0
        ), \
            f"There exists one non-stopped time record of opcode: {opcode}"

        self._maybe_sync(barrier)

        if use_cuda_event:
            # Use CUDA event to measure time cost
            # This directly measures the execution time of all kernels from the target operation
            time_record = TimeRecord(
                opcode=opcode,
                start_event=torch.cuda.Event(enable_timing=True),
                stop_event=torch.cuda.Event(enable_timing=True),
            )
            time_record.start_event.record()
        else:
            # Use Python built-in timer to measure time cost
            # This is more suitable for measuring the end-to-end time including API call, kernel 
            # lanuch and kernel execution.
            start_t=time.time()
            time_record = TimeRecord(opcode=opcode, start_t=start_t)

        if opcode in self._records:
            self._records[opcode].append(time_record)
        else:
            self._records[opcode] = [time_record]
    
    def stop(self, opcode: str, use_cuda_event: bool = False, barrier: bool = False):
        """ Stop a timer operation. """

        assert opcode in self._records, f"Undefined opcode in the timer: {opcode}"
        assert sum([int(_r.elapsed is None) for _r in self._records[opcode]]) == 1, \
            f"Must be exactly only one non-stopped time record of opcode: {opcode}"
        assert self._records[opcode][-1].elapsed is None, \
            "The last started record should be not stopped."

        self._maybe_sync(barrier)

        if use_cuda_event:
            # Use CUDA event to measure time cost
            assert isinstance(self._records[opcode][-1].stop_event, torch.cuda.Event), \
                f"The CUDA event to stop timing is not set on the last started record: " + \
                f"{self._records[opcode][-1].stop_event.__class__}"
            
            self._records[opcode][-1].stop_event.record()
            # Synchronize
            # Otherwise, the following error may occur:
            # `RuntimeError: CUDA error: device not ready. CUDA kernel errors might be asynchronously 
            # reported at some other API call, so the stacktrace below might be incorrect`.
            torch.cuda.synchronize()
            # Get elapsed time
            self._records[opcode][-1].elapsed = self._records[opcode][-1].start_event.elapsed_time(
                end_event=self._records[opcode][-1].stop_event,
            ) # Record in milsecond
            # Clear events
            self._records[opcode][-1].start_event = None
            self._records[opcode][-1].stop_event = None
        else:
            # Use Python built-in timer to measure time cost
            stop_t = time.time()
            self._records[opcode][-1].stop_t = stop_t
            self._records[opcode][-1].elapsed = (stop_t - self._records[opcode][-1].start_t)
