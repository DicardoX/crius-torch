#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Offline profiler to collect operator-level performance metrics ahead of time. """

import os
from typing import (Any, Dict, List, Optional, Tuple, Union, Callable)
from dataclasses import dataclass, field
import pickle
import time
import gc

import torch
from torch import nn

from crius.offline.offline_recorder import OfflineDataRecorder
from crius.global_envs import logger

__all__ = [
    "OfflineProfileConfigs",
    "CriusOfflineDataManager",
]


@dataclass
class OfflineProfileConfigs:
    """ Configuration for offline profiling. """

    min_batch_size: int = 1
    """ Minimum batch size for profiling. """

    max_batch_size: int = 32
    """ Maximum batch size for profiling. """

    batch_size_step: Union[int, str] = "power-of-2"
    """ Step size to increment batch size during profiling. """

    min_seq_len: int = 1
    """ Minimum sequence length for profiling. """

    max_seq_len: int = 1024
    """ Maximum sequence length for profiling. """

    seq_len_step: Union[int, str] = "power-of-2"
    """ Step size to increment sequence length during profiling. """

    dtype: torch.dtype = torch.float16
    """ Data type for profiling. Default is float16. """

    device: str = "cuda:0"
    """ Device to run the profiling on. Default is 'cuda:0'. """

    num_warmup_steps: int = 5
    """ Number of warmup steps before profiling. """

    num_profile_steps: int = 10
    """ Number of profiling steps to collect performance metrics. """

    use_torch_empty_to_init_weight: bool = True
    """ Use `torch.empty` to initialize weight tensors, instead of `torch.randn()`. 
    This can reduce the latency of GEMM-like operations, because all-zero weight incurs 
    only 1 read from GPU DRAM, then making other reads from the faster L2 cache."""

    use_transposed_weight: bool = True
    """ Use transposed weight tensor (then call weight.t() in the operator), which is a 
    common practice in Megatron for GEMM operators."""

    force_overwrite: bool = False
    """ Forcibly overwrite all profiled data and re-run. """

    # GEMM
    gemm_op_func: Callable = torch.matmul

    gemm_left_op_hidden_dims: List[int] = None
    """ Candidate hidden dimensions of the left operand in GEMM operations. """

    gemm_right_op_hidden_dims: List[int] = None
    """ Candidate hidden dimensions of the right operand in GEMM operations. """

    gemm_layouts: List[str] = field(
        default_factory=lambda: [
            # "b:s:h1,h1:h2",
            "s:b:h1,h1:h2",     # Used in Megatron
            "s:b:h2,h2:h1",
            "h1:s*b,s*b:h2",
        ]
    )
    """ Candidate layouts of GEMM operations. """

    # BMM
    bmm_op_func: Callable = torch.bmm

    bmm_op_hidden_dims: List[int] = None
    """ Candidate hidden dimensions of the operand in BMM operations. """  

    bmm_layouts: List[str] = field(
        default_factory=lambda: [
            "b:s:h1,b:h1:s", 
            "b:s:s,b:s:h1",
            "b:h1:s,b:s:s",
        ]
    )
    """ Candidate layouts of BMM operations. """ 

    num_attn_heads_per_rank: List[int] = field(
        default_factory=lambda: [2**i for i in range(0,6)]
    )
    """ Candidate number of attention heads per rank, useful for bmm and baddbmm 
    as they accumulate batch dimension as `bs * num_attn_heads_per_rank`. """

    # BADDBMM
    baddbmm_op_func: Callable = torch.baddbmm
    
    baddbmm_op_hidden_dims: List[int] = None
    """ Candidate hidden dimensions of the operand in BADDBMM operations. """

    baddbmm_layouts: List[str] = field(
        default_factory=lambda: [
            "b:s:s,b:s:h1,b:h1:s", 
        ]
    )
    """ Candidate layouts of BADDBMM operations. """

    # Communication
    allreduce_op_func: Callable = torch.distributed.all_reduce

    comm_data_sizes: List[int] = field(
        default_factory=lambda: [32*1024*4096, ]
    )
    """ Communication data sizes. """

    comm_group_size: int = 1
    """ Number of ranks in this communication group. """

    def __post_init__(self):
        # Default dimensions
        # TODO(chunyu): Maybe unify with a single attribute like `gemm_like_op_hidden_dim`
        if self.gemm_left_op_hidden_dims is None:
            self.gemm_left_op_hidden_dims = [128, 256, 512, 1024]
        if self.gemm_right_op_hidden_dims is None:
            self.gemm_right_op_hidden_dims = [128, 256, 512, 1024]
        if self.bmm_op_hidden_dims is None:
            self.bmm_op_hidden_dims = [128, 256, 512, 1024]
        if self.baddbmm_op_hidden_dims is None:
            self.baddbmm_op_hidden_dims = [128, 256, 512, 1024]

        assert isinstance(self.batch_size_step, (int, str)), "batch_size_step must be an integer or 'power-of-2'."
        assert isinstance(self.seq_len_step, (int, str)), "seq_len_step must be an integer or 'power-of-2'."


class CriusOfflineDataManager:
    """ Crius operator-level offline data manager. 
    
    This module collects (1) latencies and (2) memory footprint of computation (e.g., gemm) 
    and communication (all-reduce) operators.

    The data is organized as a hash map, where the key is generated by `OfflineDataRecorder.get_query_key()`.

    Args:
        to_store_path: Path to store the collected profiling data.
        data_filename: Name of the file to store the profiling data in pickle format.
        profile_configs: Configuration for offline profiling, including batch size, sequence length,
    """

    def __init__(
        self, 
        profile_configs: OfflineProfileConfigs,
        to_store_path: str = "./crius/offline/", 
        data_filename: str = "prof_data.pkl",
    ) -> None:
    
        self.configs = profile_configs
        self.recorder = OfflineDataRecorder(
            to_store_path=to_store_path,
            data_filename=data_filename,
            force_overwrite=profile_configs.force_overwrite,
        )

    def profile(self, opcode: str = "gemm") -> None:
        """ Profile entrypoint. """

        if opcode in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
            return self._profile_communication(opcode)
        
        num_warmup_steps = self.configs.num_warmup_steps
        num_profile_steps = self.configs.num_profile_steps
        device = torch.device(self.configs.device)
        dtype = self.configs.dtype
        op_func, dims_collection = None, []
        if self.configs.batch_size_step == "power-of-2":
            batch_sizes = [
                2**i for i in range(int(self.configs.min_batch_size).bit_length() - 1, 
                                    int(self.configs.max_batch_size).bit_length())
            ]
        else:
            raise ValueError(f"Invalid batch size step: {self.configs.batch_size_step}")
        
        if self.configs.seq_len_step == "power-of-2":
            seq_lens = [
                2**i for i in range(int(self.configs.min_seq_len).bit_length() - 1, 
                                    int(self.configs.max_seq_len).bit_length())
            ]
        else:
            raise ValueError(f"Invalid sequence length step: {self.configs.seq_len_step}")
        
        if opcode == "gemm":
            op_func = self.configs.gemm_op_func
            dims_collection = [
                self.configs.gemm_left_op_hidden_dims,
                self.configs.gemm_right_op_hidden_dims,
            ]
            profile_func = self._profile_all_gemm
        elif opcode == "bmm":
            op_func = self.configs.bmm_op_func
            dims_collection = [
                self.configs.bmm_op_hidden_dims,
            ]
            profile_func = self._profile_all_bmm
        elif opcode == "baddbmm":
            op_func = self.configs.baddbmm_op_func
            dims_collection = [
                self.configs.baddbmm_op_hidden_dims,
            ]
            profile_func = self._profile_all_baddbmm
        else:
            raise ValueError(f"Invalid opcode: {opcode}")
        
        args = (
            op_func, batch_sizes, seq_lens, dtype, device,
        )
        profile_func(*args)
        self.recorder.store()
    
    def _benchmark_one_op(
        self, opcode: str, op_func: Callable, shapes: List[tuple], num_warmup_steps: int, 
        num_profile_steps: int, dtype, device,
    ):
        """ Benchmark one operator under specific configurations. """
        if opcode in ("gemm", "bmm"):
            left_op = torch.randn(shapes[0], dtype=dtype, device=device)
            right_op = torch.randn(shapes[1], dtype=dtype, device=device) \
                if not self.configs.use_torch_empty_to_init_weight else \
                torch.empty(shapes[1], dtype=dtype, device=device)
            result_op = torch.empty(shapes[2], dtype=dtype, device=device)
            if opcode == "gemm":
                args = (left_op, right_op, ) if (
                    not self.configs.use_transposed_weight or shapes[0][-1] == shapes[1][0]
                ) else (left_op, right_op.t(), )    # Transpose if shape mismatched
            elif opcode == "bmm":
                # FIXME(chunyu): Currently we ignore the transpose(0, 1) of bmm because it 
                #                causes negligible latency difference.
                # For bmm, the torh.empty init is applied only when the value tensor is zero-like,
                # i.e., the V projection tensor is empty.
                args = (left_op, right_op, )
        
        elif opcode == "baddbmm":
            # FIXME(chunyu): Currently we ignore the transpose(0, 1) of baddbmm because it 
            #                causes negligible latency difference.
            # FIXME(chunyu): The measured latency is a little bit longer than the ground truth...
            left_op = torch.randn(shapes[0], dtype=dtype, device=device) \
                if not self.configs.use_torch_empty_to_init_weight else \
                torch.empty(shapes[0], dtype=dtype, device=device)
            mid_op = torch.randn(shapes[1], dtype=dtype, device=device) \
                if not self.configs.use_torch_empty_to_init_weight else \
                torch.empty(shapes[1], dtype=dtype, device=device)
            right_op = torch.randn(shapes[2], dtype=dtype, device=device) \
                if not self.configs.use_torch_empty_to_init_weight else \
                torch.empty(shapes[2], dtype=dtype, device=device)
            result_op = torch.zeros(shapes[3], dtype=dtype, device=device)
            args = (left_op, mid_op, right_op, )
        
        elif opcode in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
            tensor = torch.randn(shapes[0], dtype=dtype).cuda()
            if opcode == "all_reduce":
                args = (tensor, torch.distributed.ReduceOp.SUM, )
        
        else:
            raise ValueError(f"Invalid opcode: {opcode}")

        def __exec():
            if opcode in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
                op_func(*args)
            else:
                result_op = op_func(*args)

        # Warmup
        for _ in range(num_warmup_steps):
            __exec()
        torch.cuda.synchronize()

        # Timer events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark
        start_event.record()
        for _ in range(num_profile_steps):
            __exec()
        end_event.record()
        torch.cuda.synchronize()

        op_latency = start_event.elapsed_time(end_event) / num_profile_steps
        if opcode in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
            op_latency = torch.tensor([op_latency, ], device=device, dtype=dtype)
            torch.distributed.all_reduce(op_latency, op=torch.distributed.ReduceOp.MAX)
            op_latency = op_latency.item()    # Use the maximum latency across all ranks.

        return op_latency

    def _profile_all_gemm(
        self, op_func: Callable, batch_sizes: List[int], seq_lens: List[int], dtype, device, 
    ) -> None:
        for bs in batch_sizes:
            for seq_len in seq_lens:
                for left_hidden_dim in self.configs.gemm_left_op_hidden_dims:
                    for right_hidden_dim in self.configs.gemm_right_op_hidden_dims:
                        for layout in self.configs.gemm_layouts:
                            # if layout == "b:s:h1,h1:h2":
                            #     shapes = [
                            #         (bs, seq_len, left_hidden_dim),
                            #         (left_hidden_dim, right_hidden_dim),
                            #         (bs, seq_len, right_hidden_dim),
                            #     ]
                            if layout == "s:b:h1,h1:h2":
                                shapes = [
                                    (seq_len, bs, left_hidden_dim),
                                    (left_hidden_dim, right_hidden_dim),
                                    (seq_len, bs, right_hidden_dim),
                                ] if not self.configs.use_transposed_weight else [
                                    (seq_len, bs, left_hidden_dim),
                                    (right_hidden_dim, left_hidden_dim),
                                    (seq_len, bs, right_hidden_dim),
                                ]
                            elif layout == "s:b:h2,h2:h1":
                                shapes = [
                                    (bs, seq_len, right_hidden_dim),
                                    (right_hidden_dim, left_hidden_dim),
                                    (bs, seq_len, left_hidden_dim),
                                ]
                            elif layout == "h1:s*b,s*b:h2":
                                shapes = [
                                    (left_hidden_dim, seq_len * bs),
                                    (seq_len * bs, right_hidden_dim),
                                    (left_hidden_dim, right_hidden_dim),
                                ]
                            else:
                                raise ValueError(f"Unsupport GEMM layout: {layout}")

                            query_key = OfflineDataRecorder.get_query_key(
                                "gemm", bs, seq_len, left_hidden_dim, right_hidden_dim, layout,
                            )
                            if query_key in self.recorder.data and not self.configs.force_overwrite:
                                logger.info(f"Cached GEMM profile: layout: {layout}, batch_size: {bs}, seq_len: {seq_len}, " + 
                                            f"left_hd: {left_hidden_dim}, right_hd: {left_hidden_dim}")
                                continue
                            
                            # Reset
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            op_latency = self._benchmark_one_op(
                                opcode="gemm", op_func=op_func, shapes=shapes, num_warmup_steps=self.configs.num_warmup_steps, 
                                num_profile_steps=self.configs.num_profile_steps, dtype=dtype, device=device,
                            )
                            logger.info(f"==========> GEMM profile: layout: {layout}, batch_size: {bs}, seq_len: {seq_len}, " + 
                                        f"left_hd: {left_hidden_dim}, right_hd: {left_hidden_dim}, latency (ms): {op_latency}")
                            self.recorder.write_one_entry(query_key, op_latency, None)
    
    def _profile_all_bmm(
        self, op_func: Callable, batch_sizes: List[int], seq_lens: List[int], dtype, device, 
    ) -> None:
        for bs in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.bmm_op_hidden_dims:
                    for layout in self.configs.bmm_layouts:
                        for num_attn_heads in self.configs.num_attn_heads_per_rank:
                            if layout == "b:s:h1,b:h1:s":
                                shapes = [
                                    (bs * num_attn_heads, seq_len, hidden_dim // num_attn_heads),
                                    (bs * num_attn_heads, hidden_dim // num_attn_heads, seq_len),
                                    (bs * num_attn_heads, seq_len, seq_len),
                                ]
                            elif layout == "b:s:s,b:s:h1":
                                shapes = [
                                    (bs * num_attn_heads, seq_len, seq_len),
                                    (bs * num_attn_heads, seq_len, hidden_dim // num_attn_heads),
                                    (bs * num_attn_heads, seq_len, hidden_dim // num_attn_heads),
                                ]
                            elif layout == "b:h1:s,b:s:s":
                                shapes = [
                                    (bs * num_attn_heads, hidden_dim // num_attn_heads, seq_len),
                                    (bs * num_attn_heads, seq_len, seq_len),
                                    (bs * num_attn_heads, hidden_dim // num_attn_heads, seq_len),
                                ]
                            else:
                                raise ValueError(f"Unsupport BMM layout: {layout}")

                            query_key = OfflineDataRecorder.get_query_key(
                                "bmm", bs, seq_len, hidden_dim, None, layout, num_attn_heads,
                            )
                            if query_key in self.recorder.data and not self.configs.force_overwrite:
                                logger.info(f"Cached BMM profile: layout: {layout}, batch_size: {bs * num_attn_heads}, " + 
                                            f"seq_len: {seq_len}, hd: {hidden_dim // num_attn_heads}")
                                continue
                            
                            # Reset
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            op_latency = self._benchmark_one_op(
                                opcode="bmm", op_func=op_func, shapes=shapes, num_warmup_steps=self.configs.num_warmup_steps, 
                                num_profile_steps=self.configs.num_profile_steps, dtype=dtype, device=device,
                            )
                            logger.info(f"==========> BMM profile: layout: {layout}, batch_size: {bs * num_attn_heads}, " + 
                                        f"seq_len: {seq_len}, hd: {hidden_dim // num_attn_heads}, latency (ms): {op_latency}")
                            self.recorder.write_one_entry(query_key, op_latency, None)
    
    def _profile_all_baddbmm(
        self, op_func: Callable, batch_sizes: List[int], seq_lens: List[int], dtype, device, 
    ) -> None:
        for bs in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.baddbmm_op_hidden_dims:
                    for layout in self.configs.baddbmm_layouts:
                        for num_attn_heads in self.configs.num_attn_heads_per_rank:
                            if layout == "b:s:s,b:s:h1,b:h1:s":
                                shapes = [
                                    (bs * num_attn_heads, seq_len, seq_len),
                                    (bs * num_attn_heads, seq_len, hidden_dim // num_attn_heads),
                                    (bs * num_attn_heads, hidden_dim // num_attn_heads, seq_len),
                                    (bs * num_attn_heads, seq_len, seq_len),
                                ]
                            else:
                                raise ValueError(f"Unsupport BADDBMM layout: {layout}")

                            query_key = OfflineDataRecorder.get_query_key(
                                "baddbmm", bs, seq_len, hidden_dim, None, layout, num_attn_heads,
                            )
                            if query_key in self.recorder.data and not self.configs.force_overwrite:
                                logger.info(f"Cached baddbmm profile: layout: {layout}, batch_size: {bs * num_attn_heads}, " + 
                                            f"seq_len: {seq_len}, hd: {hidden_dim // num_attn_heads}")
                                continue
                            
                            # Reset
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            op_latency = self._benchmark_one_op(
                                opcode="baddbmm", op_func=op_func, shapes=shapes, num_warmup_steps=self.configs.num_warmup_steps, 
                                num_profile_steps=self.configs.num_profile_steps, dtype=dtype, device=device,
                            )
                            logger.info(f"==========> baddbmm profile: layout: {layout}, batch_size: {bs * num_attn_heads}, " + 
                                        f"seq_len: {seq_len}, hd: {hidden_dim // num_attn_heads}, latency (ms): {op_latency}")
                            self.recorder.write_one_entry(query_key, op_latency, None)
    
    def _profile_communication(
        self, opcode: str,
    ) -> None:
        """ Profile multi-device communication operators. """
        
        num_warmup_steps = self.configs.num_warmup_steps
        num_profile_steps = self.configs.num_profile_steps
        comm_data_sizes = self.configs.comm_data_sizes
        dtype = self.configs.dtype
        op_func = None
        if opcode == "all_reduce":
            op_func = self.configs.allreduce_op_func
        
        def __setup(world_size: int, rank: int, backend: str = "nccl"):
            """ Initialize the distributed environment. """
            torch.distributed.init_process_group(world_size=world_size, rank=rank, backend=backend)
            torch.cuda.set_device(rank)
        
        def __cleanup():
            """ Clean up the distributed environment. """
            torch.distributed.destroy_process_group()
        
        def __sync_barrier():
            """ Synchronize all processes. """
            torch.distributed.barrier()

        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])
        num_nodes = (world_size - 1) // local_world_size + 1
        __setup(world_size=world_size, rank=rank, backend="nccl")

        for tensor_size in comm_data_sizes:
            query_key = OfflineDataRecorder.get_query_key(
                opcode, None, None, None, None, None, None, num_nodes, world_size, tensor_size,
            )
            if query_key in self.recorder.data and not self.configs.force_overwrite:
                logger.info(
                    f"Cached {opcode} profile: num_nodes: {num_nodes}, group_size: {world_size}, " + \
                    f"tensor_size: {tensor_size}"
                )
                continue
            
            # Reset
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            __sync_barrier()

            op_latency = self._benchmark_one_op(
                opcode=opcode, op_func=op_func, shapes=[(tensor_size, ), ], num_warmup_steps=num_warmup_steps, 
                num_profile_steps=num_profile_steps, dtype=dtype, device=torch.cuda.current_device(),
            )

            if rank == 0:
                logger.info(
                    f"==========> {opcode} profile: num_nodes: {num_nodes}, group_size: {world_size}, " + \
                    f"tensor_size: {tensor_size}, latency (ms): {op_latency}"
                )
                self.recorder.write_one_entry(query_key, op_latency, None)
        
        __cleanup()
        if rank == 0:
            self.recorder.store()
        