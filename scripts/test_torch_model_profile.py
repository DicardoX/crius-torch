#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set
import unittest
import logging
from functools import partial
from collections import OrderedDict
import time
import numpy as np

import torch
from torch import nn
import torch.distributed as dist

from crius.loader import (ModelLoader, ShadowModelLoader)
from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan

# Megatron imports for tensor parallelism and pipeline parallelism
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule
from megatron.core.enums import ModelType


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: str = torch.float16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return self.fc2(x)


class ToyLayerTP(nn.Module):
    """Tensor parallel version of ToyLayer using Megatron's ColumnParallelLinear and RowParallelLinear."""
    
    def __init__(self, hidden_size: int, config: ModelParallelConfig, device: str, dtype: str = torch.float16) -> None:
        super().__init__()
        self.dtype = dtype
        # Column parallel: split along output dimension
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            config=config,
            init_method=torch.nn.init.xavier_uniform_,
            bias=False,
            gather_output=False,  # Don't gather output, let next layer handle distributed input
            skip_bias_add=False,
        )
        # Row parallel: split along input dimension, gather output
        self.fc2 = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            config=config,
            init_method=torch.nn.init.xavier_uniform_,
            bias=False,
            input_is_parallel=True,  # Input is already distributed from previous layer
            skip_bias_add=False,
        )
        
        # Ensure weights are in the correct dtype and device
        self.fc1.weight.data = self.fc1.weight.data.to(dtype=dtype, device=device)
        self.fc2.weight.data = self.fc2.weight.data.to(dtype=dtype, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x, _ = self.fc2(x)
        return x


class ToyModel(nn.Module):
    """ Toy PyTorch model. """

    def __init__(
            self, hidden_size: int, num_layers: int, device: str, dtype: str = torch.float16,
    ) -> None:
        """ Initialize a toy model. """

        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList([ToyLayer(hidden_size, device, dtype) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        
    def _gemm_op(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        """ Atomic GEMM operator. """
        hidden_states = self.layers[layer_index](hidden_states)
        return hidden_states

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """ Forward function. """
        hidden_states = input_
        for i in range(self.num_layers):
            hidden_states = self._gemm_op(hidden_states, i)
        return self.layer_norm(hidden_states)


class ToyModelTP(nn.Module):
    """ Tensor parallel version of ToyModel using Megatron's tensor parallel layers. """

    def __init__(
            self, hidden_size: int, num_layers: int, config: ModelParallelConfig, device: str, dtype: str = torch.float16,
    ) -> None:
        """ Initialize a tensor parallel toy model. """

        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList([ToyLayerTP(hidden_size, config, device, dtype) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        
    def _gemm_op(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        """ Atomic GEMM operator. """
        hidden_states = self.layers[layer_index](hidden_states)
        return hidden_states

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """ Forward function. """
        hidden_states = input_
        for i in range(self.num_layers):
            hidden_states = self._gemm_op(hidden_states, i)
        return self.layer_norm(hidden_states)


class ToyModelPP(MegatronModule):
    def __init__(
            self, hidden_size: int, num_layers: int, config: ModelParallelConfig, 
            device: str, dtype: str = torch.float16, pre_process: bool = True, post_process: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.device = device
        self.pre_process = pre_process
        self.post_process = post_process
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        
        layers_per_stage = num_layers // pp_world_size
        start_layer = pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        if pp_rank == pp_world_size - 1:
            end_layer = num_layers
            
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers_this_stage = end_layer - start_layer
        
        self.layers = nn.ModuleList([
            ToyLayer(hidden_size, device, dtype) for _ in range(self.num_layers_this_stage)
        ])
        
        if self.post_process:
            self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        else:
            self.layer_norm = None
            
        self.input_tensor = None
        
    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        
    def _gemm_op(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        hidden_states = self.layers[layer_index](hidden_states)
        return hidden_states

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if parallel_state.is_pipeline_first_stage():
            hidden_states = input_
        else:
            if self.input_tensor is not None:
                if isinstance(self.input_tensor, list):
                    hidden_states = self.input_tensor[0]
                else:
                    hidden_states = self.input_tensor
            else:
                raise RuntimeError("input_tensor is None for non-first pipeline stage")
            
        for i in range(self.num_layers_this_stage):
            hidden_states = self._gemm_op(hidden_states, i)
            
        if self.post_process and self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        return hidden_states


class ToyModelTPPP(MegatronModule):
    def __init__(
            self, hidden_size: int, num_layers: int, config: ModelParallelConfig, 
            device: str, dtype: str = torch.float16, pre_process: bool = True, post_process: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.device = device
        self.pre_process = pre_process
        self.post_process = post_process
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        
        layers_per_stage = num_layers // pp_world_size
        start_layer = pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        if pp_rank == pp_world_size - 1:
            end_layer = num_layers
            
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers_this_stage = end_layer - start_layer
        
        self.layers = nn.ModuleList([
            ToyLayerTP(hidden_size, config, device, dtype) for _ in range(self.num_layers_this_stage)
        ])
        
        if self.post_process:
            self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        else:
            self.layer_norm = None
            
        self.input_tensor = None
        
    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        
    def _gemm_op(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        hidden_states = self.layers[layer_index](hidden_states)
        return hidden_states

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if parallel_state.is_pipeline_first_stage():
            hidden_states = input_
        else:
            if self.input_tensor is not None:
                if isinstance(self.input_tensor, list):
                    hidden_states = self.input_tensor[0]
                else:
                    hidden_states = self.input_tensor
            else:
                raise RuntimeError("input_tensor is None for non-first pipeline stage")
            
        for i in range(self.num_layers_this_stage):
            hidden_states = self._gemm_op(hidden_states, i)
            
        if self.post_process and self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        return hidden_states


def initialize_distributed_and_tp(tensor_parallel_size: int = 1):
    """Initialize distributed training and tensor parallelism."""
    
    if not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=int(os.environ.get('RANK', 0))
        )
    
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=1,
        )
    
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)
    
    return parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()


def initialize_distributed_and_pp(pipeline_parallel_size: int = 1, tensor_parallel_size: int = 1):
    if not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=int(os.environ.get('RANK', 0))
        )
    
    
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
    
    
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)
    
    return (parallel_state.get_pipeline_model_parallel_rank(), 
            parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_tensor_model_parallel_rank(), 
            parallel_state.get_tensor_model_parallel_world_size())


def model_provider(hidden_size: int, num_layers: int, device: str, dtype: str) -> nn.Module:
    """ Model provider function. """
    return ToyModel(hidden_size, num_layers, device, dtype=dtype)


def model_provider_tp(hidden_size: int, num_layers: int, device: str, dtype: str, config: ModelParallelConfig) -> nn.Module:
    """ Tensor parallel model provider function. """
    return ToyModelTP(hidden_size, num_layers, config, device, dtype=dtype)


def model_provider_pp(hidden_size: int, num_layers: int, device: str, dtype: str, config: ModelParallelConfig, 
                      pre_process: bool = True, post_process: bool = True) -> nn.Module:
    """ Pipeline parallel model provider function. """
    return ToyModelPP(hidden_size, num_layers, config, device, dtype=dtype, 
                      pre_process=pre_process, post_process=post_process)


def model_provider_tp_pp(hidden_size: int, num_layers: int, device: str, dtype: str, config: ModelParallelConfig, 
                         pre_process: bool = True, post_process: bool = True) -> nn.Module:
    """ Tensor + Pipeline parallel model provider function. """
    return ToyModelTPPP(hidden_size, num_layers, config, device, dtype=dtype, 
                        pre_process=pre_process, post_process=post_process)


class TorchModelProfileTest(unittest.TestCase):
    """ Basic torch model disaggregated profiling test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    @unittest.skip("Pass.")
    def test_torch_model_offline_profiling(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=1,
            max_batch_size=4,
            batch_size_step="power-of-2",
            min_seq_len=32,
            max_seq_len=2048,
            seq_len_step="power-of-2",
            gemm_left_op_hidden_dims=[4096, ],
            gemm_right_op_hidden_dims=[4096, ],
            dtype=torch.float16,
            device="cuda",
            num_warmup_steps=5,
            num_profile_steps=10,
            force_overwrite=True,
        )
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        
        offline_profiler.profile_all_gemm()

    # @unittest.skip("Pass.")
    def test_torch_model_profiling(self):
        """ Profile a naive torch model. """
        
        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = "cuda"
        enable_torch_profiler = False

        
        # _model_loader_class = ModelLoader
        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
                "dtype": dtype,
            },
            # device_str=device,
            fake_device_str=device,
        )
        model = model_loader.load()
        
        
        data = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]

        
        # TODO(chunyu): Currently we manually specify the mappings
        op_to_layer_mapping = {
            'L__self___layers_0_fc1': 0,
            'L__self___layers_0_fc2': 0,
            'L__self___layers_1_fc1': 1,
            'L__self___layers_1_fc2': 1,
            'L__self___layers_2_fc1': 2,
            'L__self___layers_2_fc2': 2,
            'L__self___layers_3_fc1': 3,
            'L__self___layers_3_fc2': 3,
            'L__self___layer_norm': 3,
        }
        layer_to_stage_mapping = [[0, 1, 2, 3]]
        parallelism_plan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=1,
            num_microbatches=num_micro_batches,
            tensor_parallel_degree=1,
            data_parallel_degree=1,
            op_to_layer_mapping=op_to_layer_mapping,
            layer_to_stage_mapping=layer_to_stage_mapping,
        )
        # Profile
        profiler = CriusDistProfiler(
            profiled_data_path='./crius/offline/prof_data.pkl',
        )
        e2e_latency = profiler.profile(
            model=model,
            base_input_args=(data[0], ),
            parallelism_plan=parallelism_plan,
            only_forward=True,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        ################# Real profiling #################
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
                "dtype": dtype,
            },
            device_str=device,
        )
        model = model_loader.load()

        # Warmup
        for _ in range(100):
            with torch.no_grad():
                _ = model(data[0])
        torch.cuda.synchronize(device)

        if not enable_torch_profiler:     
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            # Measure
            start_event.record()
            for _ in range(1000):
                with torch.no_grad():
                    _ = model(data[0])
            end_event.record()    

            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / 1000     
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
            
            print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")
        
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                # on_trace_ready=lambda x: x.export_chrome_trace('/app/torch_exec_trace.json'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for _ in range(1):
                    with torch.no_grad():
                        _ = model(data[0])
                prof.step()
        
            prof.export_chrome_trace('/app/torch_exec_trace.json')

    # @unittest.skip("Pass.")
    def test_torch_model_tensor_parallel_profiling(self):
        """ Profile a tensor parallel torch model using Megatron. """
        
        # Initialize distributed and tensor parallelism
        tensor_parallel_size = int(os.environ.get('WORLD_SIZE', 2))  # Default to 2 for TP
        tp_rank, tp_world_size = initialize_distributed_and_tp(tensor_parallel_size)
        
        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = f"cuda:{torch.cuda.current_device()}"
        
        # Create ModelParallelConfig for Megatron
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp_world_size,
            pipeline_model_parallel_size=1,
        )
        
        print(f"[Rank {tp_rank}] Initializing tensor parallel model with TP size: {tp_world_size}")
        
        # Create tensor parallel model
        model = model_provider_tp(hidden_size, num_layers, device, dtype, config)
        
        # Inputs (same across all TP ranks)
        torch.manual_seed(1234)  # Ensure same input across ranks
        data = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]
        
        # Warmup
        print(f"[Rank {tp_rank}] Starting warmup...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(data[0])
        torch.cuda.synchronize(device)
        
        # Measure tensor parallel performance
        print(f"[Rank {tp_rank}] Starting measurement...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        
        start_event.record()
        for _ in range(100):
            with torch.no_grad():
                output = model(data[0])
        end_event.record()    
        
        end_event.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 100
        
        if tp_rank == 0:
            print(f"========> Tensor Parallel (TP={tp_world_size}) end-to-end latency: {avg_latency_ms:.3f} ms")
            print(f"========> Output shape: {output.shape}")
        
        # Cleanup
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    # @unittest.skip("Pass.")
    def test_torch_model_data_parallel_profiling(self):
        """ Profile a data parallel torch model using torchrun. """
        
        # Initialize distributed training for data parallelism
        if not dist.is_initialized():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            rank = int(os.environ.get('RANK', 0))
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            rank = int(os.environ.get('RANK', 0))
        
        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = f"cuda:{local_rank}"
        
        print(f"[Rank {rank}] Initializing data parallel model with DP world size: {world_size}")
        
        # Create regular model and wrap with DDP
        model = model_provider(hidden_size, num_layers, device, dtype)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        
        # Each rank gets a different portion of the global batch for data parallelism
        local_batch_size = global_bs // world_size
        torch.manual_seed(1234 + rank)  # Different seed per rank for different data
        data = [
            torch.randn(local_batch_size, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]
        
        # Warmup
        print(f"[Rank {rank}] Starting warmup...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(data[0])
        torch.cuda.synchronize(device)
        
        # Measure data parallel performance
        print(f"[Rank {rank}] Starting measurement...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        
        start_event.record()
        for _ in range(100):
            with torch.no_grad():
                output = model(data[0])
        end_event.record()    
        
        end_event.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 100
        
        if rank == 0:
            print(f"========> Data Parallel (DP={world_size}) end-to-end latency: {avg_latency_ms:.3f} ms")
            print(f"========> Local batch size per rank: {local_batch_size}")
            print(f"========> Output shape: {output.shape}")
        
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

    # @unittest.skip("Pass.")
    def test_torch_model_pipeline_parallel_profiling(self):
        
        pipeline_parallel_size = int(os.environ.get('WORLD_SIZE', 2))  # Default to 2 for PP
        pp_rank, pp_world_size, tp_rank, tp_world_size = initialize_distributed_and_pp(
            pipeline_parallel_size=pipeline_parallel_size, tensor_parallel_size=1
        )
        
        global_bs = 8
        seq_len = 2048
        num_micro_batches = 4
        hidden_size = 4096
        num_layers = 8
        dtype = torch.float16
        device = f"cuda:{torch.cuda.current_device()}"
        
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp_world_size,
            pipeline_model_parallel_size=pp_world_size,
            pipeline_dtype=dtype,
        )
        config.hidden_size = hidden_size
        config.fp8 = None
        config.sequence_parallel = False
        
        print(f"[PP Rank {pp_rank}] Initializing pipeline parallel model with PP size: {pp_world_size}")
        
        pre_process = (pp_rank == 0)
        post_process = (pp_rank == pp_world_size - 1)
        
        model = model_provider_pp(hidden_size, num_layers, device, dtype, config, 
                                  pre_process=pre_process, post_process=post_process)
        
        print(f"[PP Rank {pp_rank}] Model created with layers {model.start_layer}-{model.end_layer-1}")
        
        def forward_step_func(data_iterator, model):
            data = next(data_iterator)
            output = model(data)
            
            def loss_func(output_tensor):
                if parallel_state.is_pipeline_last_stage():
                    target = torch.zeros_like(output_tensor)
                    loss = torch.nn.functional.mse_loss(output_tensor, target)
                    return loss, {'loss': loss}
                else:
                    return output_tensor, {}
                    
            return output, loss_func
        
        torch.manual_seed(1234)  # Ensure same input across ranks
        data_list = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
            for _ in range(num_micro_batches)
        ]
        data_iterator = iter(data_list)
        
        forward_backward_func = get_forward_backward_func()
        
        # Warmup
        print(f"[PP Rank {pp_rank}] Starting warmup...")
        for _ in range(5):
            data_iterator = iter(data_list)
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_micro_batches,
                seq_length=seq_len,
                micro_batch_size=global_bs // num_micro_batches,
                forward_only=True,
            )
        torch.cuda.synchronize(device)
        
        print(f"[PP Rank {pp_rank}] Starting measurement...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        
        start_event.record()
        for _ in range(10):
            data_iterator = iter(data_list)
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_micro_batches,
                seq_length=seq_len,
                micro_batch_size=global_bs // num_micro_batches,
                forward_only=True,
            )
        end_event.record()    
        
        end_event.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 10
        
        if pp_rank == 0:
            print(f"========> Pipeline Parallel (PP={pp_world_size}) end-to-end latency: {avg_latency_ms:.3f} ms")
            print(f"========> Number of microbatches: {num_micro_batches}")
            print(f"========> Layers per stage: {num_layers // pp_world_size}")
        
        # Cleanup
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    # @unittest.skip("Pass.")
    def test_torch_model_hybrid_tp_pp_profiling(self):
        """ Profile a hybrid parallel torch model using both TP and PP. """
        world_size = int(os.environ.get('WORLD_SIZE', 4))
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        tensor_parallel_size = 2  # TP size
        pipeline_parallel_size = world_size // tensor_parallel_size  # PP size
        
        assert world_size % tensor_parallel_size == 0, f"World size {world_size} must be divisible by TP size {tensor_parallel_size}"
        
        pp_rank, pp_world_size, tp_rank, tp_world_size = initialize_distributed_and_pp(
            pipeline_parallel_size=pipeline_parallel_size, 
            tensor_parallel_size=tensor_parallel_size
        )
        
        global_bs = 8
        seq_len = 2048
        num_micro_batches = 4
        hidden_size = 4096
        num_layers = 8
        dtype = torch.float16
        device = f"cuda:{local_rank}"
        
        print(f"[Global Rank {rank}] Hybrid TP+PP: PP_rank={pp_rank}/{pp_world_size}, TP_rank={tp_rank}/{tp_world_size}")
        
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp_world_size,
            pipeline_model_parallel_size=pp_world_size,
            pipeline_dtype=dtype,
        )
        config.hidden_size = hidden_size
        config.fp8 = None
        config.sequence_parallel = False
        
        pre_process = (pp_rank == 0)
        post_process = (pp_rank == pp_world_size - 1)
        
        model = model_provider_tp_pp(hidden_size, num_layers, device, dtype, config, 
                                     pre_process=pre_process, post_process=post_process)
        
        print(f"[Global Rank {rank}] Model created with layers {model.start_layer}-{model.end_layer-1}")
        
        def forward_step_func(data_iterator, model):
            data = next(data_iterator)
            output = model(data)
            
            def loss_func(output_tensor):
                if parallel_state.is_pipeline_last_stage():
                    target = torch.zeros_like(output_tensor)
                    loss = torch.nn.functional.mse_loss(output_tensor, target)
                    return loss, {'loss': loss}
                else:
                    return output_tensor, {}
                    
            return output, loss_func
        
        torch.manual_seed(1234)  # Ensure same input across ranks
        data_list = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
            for _ in range(num_micro_batches)
        ]
        
        forward_backward_func = get_forward_backward_func()
        
        # Warmup
        print(f"[Global Rank {rank}] Starting warmup...")
        for _ in range(5):
            data_iterator = iter(data_list)
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_micro_batches,
                seq_length=seq_len,
                micro_batch_size=global_bs // num_micro_batches,
                forward_only=True,
            )
        torch.cuda.synchronize(device)
        
        print(f"[Global Rank {rank}] Starting measurement...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        
        start_event.record()
        for _ in range(10):
            data_iterator = iter(data_list)
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_micro_batches,
                seq_length=seq_len,
                micro_batch_size=global_bs // num_micro_batches,
                forward_only=True,
            )
        end_event.record()    
        
        end_event.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 10
        
        if rank == 0:
            print(f"========> Hybrid Parallel (TP={tensor_parallel_size}, PP={pipeline_parallel_size}) end-to-end latency: {avg_latency_ms:.3f} ms")
            print(f"========> Number of microbatches: {num_micro_batches}")
            print(f"========> Layers per PP stage: {num_layers // pipeline_parallel_size}")
            print(f"========> TP degree per stage: {tensor_parallel_size}")
        
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()

"""
# Tensor Parallel (TP)
conda activate crius && cd /app && torchrun --nproc-per-node=2 scripts/test_torch_model_profile.py TorchModelProfileTest.test_torch_model_tensor_parallel_profiling

# Data Parallel (DP)
conda activate crius && cd /app && torchrun --nproc-per-node=2 scripts/test_torch_model_profile.py TorchModelProfileTest.test_torch_model_data_parallel_profiling

# Pipeline Parallel (PP)
conda activate crius && cd /app && torchrun --nproc-per-node=2 scripts/test_torch_model_profile.py TorchModelProfileTest.test_torch_model_pipeline_parallel_profiling

# Hybrid TP + PP
conda activate crius && cd /app && torchrun --nproc-per-node=4 scripts/test_torch_model_profile.py TorchModelProfileTest.test_torch_model_hybrid_tp_pp_profiling
"""