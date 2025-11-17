#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set, Any
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
from crius.core.grid_planner.planner import CriusParallelPlanner
from crius.model_parallel.utils import ParallelismPlan
from crius.global_envs import global_configs, logger

# Megatron imports for tensor parallelism and pipeline parallelism
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule
from megatron.core.enums import ModelType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

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

def initialize_distributed_and_3d(pipeline_parallel_size: int = 1, tensor_parallel_size: int = 1):
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
            parallel_state.get_tensor_model_parallel_world_size(),
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_world_size())

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
            # min_batch_size=1,
            min_batch_size=128,
            max_batch_size=128,
            batch_size_step="power-of-2",
            min_seq_len=128,
            max_seq_len=128,
            # max_seq_len=128,
            seq_len_step="power-of-2",
            gemm_left_op_hidden_dims =[128, 512],
            # gemm_left_op_hidden_dims =[128, 256, 5000],
            gemm_right_op_hidden_dims=[128, 10000],
            # gemm_right_op_hidden_dims =[128, 256, 5000],
            reduce_sum_left_op_hidden_dims =[128,],
            reduce_sum_right_op_hidden_dims=[10000],
            bmm_op_hidden_dims =[128],
            # bmm_right_op_hidden_dims=[512, 1024, 1536, 2048, 25000],
            baddbmm_op_hidden_dims =[128],
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
        
        # offline_profiler.profile_all_gemm()
        # offline_profiler.profile_all_baddbmm()
        # offline_profiler.profile_all_reduce_sum()
        # offline_profiler.profile_all_bmm()
        offline_profiler.profile_all_comm()

    @unittest.skip("Pass.")
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

        # Model
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
        
        # Inputs
        data = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]

        # Parallelism plan
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
    def test_megatron_model_profiling(self):
        """ Profile a Megatron model. """

        # from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.module import MegatronModule
        from megatron.core import mpu
        # from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from crius.train_backend.megatron.utils import initialize_distributed
        from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
        from crius.train_backend.megatron.transformer_layer import TransformerLayer
        from crius.train_backend.megatron.gpt_model import GPTModel

        class MegatronToyLayer(MegatronModule):
            # TODO(chunyu): Replace nn.Linear with Megatron's tensor parallelized linear layer.
            
            def __init__(self, hidden_size: int, device: str, dtype: str = torch.float16) -> None:
                super().__init__(config=None)
                self.fc1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
                self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = self.fc2(x)
                torch.distributed.all_reduce(x, group=mpu.get_tensor_model_parallel_group())
                return x
        
        class ToyLayerTP(nn.Module):
            """Tensor parallel version of ToyLayer using Megatron's ColumnParallelLinear and RowParallelLinear."""
            
            def __init__(
                self, hidden_size: int, config: TransformerConfig, device: str, dtype: str = torch.float16,
            ) -> None:
                super().__init__()
                self.dtype = dtype
                # Column parallel: split along output dimension
                self.fc1 = ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    tp_size=config.tensor_model_parallel_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                # Row parallel: split along input dimension, gather output
                self.fc2 = RowParallelLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    tp_size=config.tensor_model_parallel_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                
                # Ensure weights are in the correct dtype and device
                self.fc1.weight.data = self.fc1.weight.data.to(dtype=dtype, device=device)
                self.fc2.weight.data = self.fc2.weight.data.to(dtype=dtype, device=device)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x, _ = self.fc1(x)
                x, _ = self.fc2(x)
                return x

        class MegatronToyModel(MegatronModule):
            """ Toy Megatron model. """

            def __init__(self, hidden_size: int, num_layers: int, device: str, dtype: str = torch.float16, megatron_config: Any = None) -> None:
                super().__init__(config=megatron_config)
                self.device = device
                self.num_layers = num_layers
                # self.layers = nn.ModuleList([MegatronToyLayer(hidden_size, device, dtype) for _ in range(num_layers)])
                self.layers = nn.ModuleList([ToyLayerTP(hidden_size, megatron_config, device, dtype) for _ in range(num_layers)])
                self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)

            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                hidden_states = input_
                for i in range(self.num_layers):
                    hidden_states = self.layers[i](hidden_states)
                return self.layer_norm(hidden_states)
        
        def model_provider(
            hidden_size: int, 
            num_layers: int, 
            max_seq_len: int, 
            vocab_size: int, 
            device: str, 
            dtype: str, 
            megatron_config: Any = None, 
            use_toy: bool = False, 
            _pre_process: bool = False, 
            _post_process: bool = False
        ) -> nn.Module:
            """ Model provider function. """
            if use_toy:
                return MegatronToyModel(hidden_size, num_layers, device, dtype=dtype, megatron_config=megatron_config)
            else:
                return GPTModel(megatron_config, vocab_size, max_seq_len, device=device, pre_process=_pre_process, post_process=_post_process)

        class DummyDataset(Dataset):
            def __init__(self, num_samples=100, seq_len=128, vocab_size=1000):
                self.num_samples = num_samples
                self.seq_len = seq_len
                self.vocab_size = vocab_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
                    'position_ids': torch.arange(self.seq_len),
                    'attention_mask': torch.tril(torch.ones(self.seq_len, self.seq_len)).bool().unsqueeze(0),  # 因果mask，布尔类型
                    'labels': torch.randint(0, self.vocab_size, (self.seq_len,)),
                }

        global_bs = 4
        seq_len = 128
        num_micro_batches = 4
        hidden_size = 256
        num_layers = 1
        data_dtype = torch.long
        dtype = torch.float16
        device = "cuda"
        vocab_size = 10000
        use_toy = False
        
        global_configs.training_backend = "megatron"
        global_configs.tensor_model_parallel_size = 2
        global_configs.pipeline_model_parallel_size = 1
        global_configs.data_parallel_size = 1
        global_configs.only_forward = False
        global_configs.enable_backward_graph_generation = True

        # Parallelism plan
        # op_to_layer_mapping = {           # FIXME(chunyu): Currently this manual mapping remains legacy and not supported...
        #     'L__self___layers_0_fc1': 0,
        #     'L__self___layers_0_fc2': 0,
        #     'L__self___layers_1_fc1': 1,
        #     'L__self___layers_1_fc2': 1,
        #     'L__self___layers_2_fc1': 2,
        #     'L__self___layers_2_fc2': 2,
        #     'L__self___layers_3_fc1': 3,
        #     'L__self___layers_3_fc2': 3,
        #     'L__self___layer_norm': 3,
        # }
        # layer_to_stage_mapping = [[0, 1, 2, 3]]
        op_to_layer_mapping, layer_to_stage_mapping = None, None
        pipeline_stage_partition_method = "uniform"
        # pipeline_num_preprocess_nodes = 1
        # pipeline_num_postprocess_nodes = 2
        pipeline_use_boundary_markers = True
        pipeline_boundary_start_marker = ['getattr', 'input_layernorm', ]
        pipeline_boundary_end_marker = ['dropout', 'add', ]

        parallelism_plan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            num_microbatches=num_micro_batches,
            tensor_parallel_degree=global_configs.tensor_model_parallel_size,
            data_parallel_degree=global_configs.data_parallel_size,
            pipeline_stage_partition_method=pipeline_stage_partition_method,
            # pipeline_num_preprocess_nodes=pipeline_num_preprocess_nodes,
            # pipeline_num_postprocess_nodes=pipeline_num_postprocess_nodes,
            pipeline_use_boundary_markers=pipeline_use_boundary_markers,
            pipeline_boundary_start_marker=pipeline_boundary_start_marker,
            pipeline_boundary_end_marker=pipeline_boundary_end_marker,
            op_to_layer_mapping=op_to_layer_mapping,
            layer_to_stage_mapping=layer_to_stage_mapping,
        )

        # We manually set the communication op insertion rules
        global_configs.comm_op_insert_rules = {
            "all_reduce:tp:forward": {
                # "after_nodes": ['fc2', ],
                # "after_nodes": ['matmul', ],
                "after_markers": ['self_attention_linear_proj_weight', 'mlp_linear_fc2_weight', ],
                "after_nodes": ['contiguous', ],
                "interval": 1,
                "pass": 'forward',
                "parallelism": 'tp',
            },
            "all_reduce:dp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad_reduce_sum', ],
                "interval": 1,
                "pass": 'backward',
                "parallelism": 'dp',
            },
            "all_reduce:tp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad_reduce_sum', ],
                "skip_first_n": 4,  # Addition 1 gemm before and after transformer layer loop
                "interval": 4,      # Each transformer layer's gemm: qkv, out proj, mlp-fc1, mlp-fc2
                "pass": 'backward',
                "parallelism": 'tp',
            },
        }

        # Init distributed
        initialize_distributed(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron
        megatron_config = TransformerConfig(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=max(1, global_configs.tensor_model_parallel_size),
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=True,
            # num_layers=4,
            # hidden_size=256,
            # num_attention_heads=4,
            # pipeline_model_parallel_size=2,
            # tensor_model_parallel_size=1,
            # pipeline_dtype=torch.float16,  # 必须指定
            # params_dtype=torch.float16,
        )
        setattr(megatron_config, "max_sequence_length", seq_len)
        setattr(megatron_config, "vocab_size", vocab_size)
        setattr(megatron_config, "microbatch_size", global_bs // num_micro_batches)

        # Model
        # _model_loader_class = ShadowModelLoader
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "max_seq_len": seq_len,
                "vocab_size": vocab_size,
                "device": device,
                "dtype": dtype,
                "megatron_config": megatron_config,
                "use_toy": use_toy,
                "_pre_process": False,
                "_post_process": False
            },
            device_str=device,
            # fake_device_str=device,
        )
        model = model_loader.load()
        
        # Inputs
        if use_toy:
            data = [
                torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                    for _ in range(num_micro_batches)
            ]
            input_args = (data[0], )
            input_kwargs = {}
        else:
            input_ids = torch.ones(global_bs // num_micro_batches, seq_len, dtype=data_dtype, device=device)
            position_ids = torch.arange(seq_len, dtype=data_dtype, device=device).unsqueeze(0).expand_as(input_ids)
            attention_mask = torch.ones(global_bs // num_micro_batches, 1, seq_len, seq_len, dtype=torch.bool, device=device)
            decoder_input = None
            labels = torch.ones(global_bs // num_micro_batches, seq_len, dtype=data_dtype, device=device)
            input_args = ()
            input_kwargs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "labels": labels,
            }

        profiler = CriusDistProfiler(
            profiled_cmpt_data_path='./crius/offline/prof_data.pkl',
            profiled_comm_data_path='./crius/offline/prof_comm_data.pkl',
        )
        e2e_latency = profiler.profile(
            model=model,
            parallelism_plan=parallelism_plan,
            only_forward=global_configs.only_forward,
            base_input_args=input_args,
            base_input_kwargs=input_kwargs,
            partition_pipeline_stages_func=CriusParallelPlanner.partition_pipeline_stages,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        exit(0)

        ################# Real profiling #################
        if not parallel_state.model_parallel_is_initialized():
            try: 
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
                    pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
                )
            except:
                print("Failed to set [parallel_state].")

        rank = dist.get_rank()

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        is_first = parallel_state.is_pipeline_first_stage()
        is_last = parallel_state.is_pipeline_last_stage()

        print(f"[Rank {rank}] Pipeline rank: {pipeline_rank}, "
              f"First: {is_first}, Last: {is_last}")

        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "max_seq_len": seq_len,
                "vocab_size": vocab_size,
                "device": device,
                "dtype": dtype,
                "megatron_config": megatron_config,
                "use_toy": use_toy,
                "_pre_process": is_first,
                "_post_process": is_last
            },
            device_str=device,
        )
        model = model_loader.load()

        # Inputs
        def generate_inputs(batch_size, seq_len, hidden_size, dtype, device):
            # input_ = torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=torch.float16, device=device)     
            input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device)
            return {
                # 'input_': input_,
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask
            }

        data = [generate_inputs(global_bs // num_micro_batches, seq_len, hidden_size, torch.long, device) 
                for _ in range(num_micro_batches)]

        measured_e2e_latency = 0.0

        if global_configs.pipeline_model_parallel_size <= 1:
            
            # Warmup
            for _ in range(100):
                with torch.no_grad():
                    _ = model(**data[0])
            torch.cuda.synchronize(device)
        
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  

            # Measure
            start_event.record()
            for _ in range(1000):
                with torch.no_grad():
                    _ = model(**data[0])
            end_event.record()

            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / 1000     
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        else:   
            
            # def register_debug_hooks(model):
    
            #     def hook_fn(module, input, output, layer_name):
            #         print(f"\n=== {layer_name} ===")
            #         print(f"Module type: {type(module).__name__}")
            #         print(f"Input shapes: {[inp.shape if hasattr(inp, 'shape') else str(type(inp)) for inp in input]}")
            #         print(f"Output shape: {output.shape if hasattr(output, 'shape') else str(type(output))}")
                    
            #         # 如果是特定模块，打印更多信息
            #         if hasattr(module, 'weight'):
            #             print(f"Weight shape: {module.weight.shape}")
            #         if hasattr(module, 'in_features'):
            #             print(f"in_features: {module.in_features}, out_features: {module.out_features}")
                
            #     hooks = []
            #     # 递归注册hook到所有子模块
            #     for name, module in model.named_modules():
            #         if name:  # 跳过根模块
            #             hook = module.register_forward_hook(
            #                 lambda m, i, o, name=name: hook_fn(m, i, o, name)
            #             )
            #             hooks.append(hook)
                
            #     return hooks

            # if rank == 0:
                # hooks = register_debug_hooks(model.module)  # 去掉DistributedDataParallel包装
            
            # print(model.config)
            if is_first:
                dataset = DummyDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=num_micro_batches, 
                    # collate_fn=custom_collate_fn
                )
                
                def gpu_data_iterator():
                    for batch in dataloader:
                        # print(len(batch.items()))
                        yield {k: v.cuda() for k, v in batch.items()}
                
                data_iterator = gpu_data_iterator()
            else:
                data_iterator = None

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Warmup
            print(f"[Global Rank {rank}] Starting warmup...")
            for _ in range(100):
                losses = model.module.pipeline_forward_backward(
                    data_iterator=data_iterator,
                    optimizer=optimizer,
                    num_microbatches=num_micro_batches,
                    seq_len=seq_len, 
                    forward_only=False
                )
            # torch.cuda.synchronize(device)
            dist.barrier()
            
            print(f"[Global Rank {rank}] Starting measurement...")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            
            start_event.record()
            for _ in range(100):
                losses = model.module.pipeline_forward_backward(
                    data_iterator=data_iterator,
                    optimizer=optimizer,
                    num_microbatches=num_micro_batches,
                    seq_len=seq_len, 
                    forward_only=False
                )
            end_event.record()    
            
            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / 100
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        
        print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")

    @unittest.skip("Pass.")
    def test_device_free_profiling(self):
        """ Profile a Megatron model in a device-free manner. """

        # from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.module import MegatronModule
        from megatron.core import mpu
        # from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from crius.train_backend.megatron.utils import initialize_virtual_distributed
        from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
        from crius.train_backend.megatron.transformer_layer import TransformerLayer
        from crius.train_backend.megatron.gpt_model import GPTModel

        def model_provider(
            hidden_size: int, num_layers: int, max_seq_len: int, vocab_size: int, device: str, dtype: str, 
            megatron_config: Any = None, use_toy: bool = False,
        ) -> nn.Module:
            """ Model provider function. """
            return GPTModel(megatron_config, vocab_size, max_seq_len, device=device)

        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        data_dtype = torch.long
        dtype = torch.float16
        device = "cpu"
        vocab_size = 50000
        
        global_configs.tensor_model_parallel_size = 1
        global_configs.pipeline_model_parallel_size = 2
        global_configs.data_parallel_size = 1
        global_configs.training_backend = "megatron"
        global_configs.only_forward = False
        global_configs.enable_backward_graph_generation = True
        if device == "cpu":
            logger.warning(
                "Shape propagation currently hangs in CPU mode, or prabably due to extremely slow progagation. " +
                "Currently we recommend using shadow model loader to reduce device memory consumption."
            )
            device = "cuda"
            global_configs.current_device = torch.cuda.current_device()
            global_configs.use_shadow_model_loader = True

        # Parallelism plan
        pipeline_stage_partition_method = "uniform"
        pipeline_use_boundary_markers = True
        pipeline_boundary_start_marker = ['getattr', 'input_layernorm', ]
        pipeline_boundary_end_marker = ['dropout', 'add', ]

        parallelism_plan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            num_microbatches=num_micro_batches,
            tensor_parallel_degree=global_configs.tensor_model_parallel_size,
            data_parallel_degree=global_configs.data_parallel_size,
            pipeline_stage_partition_method=pipeline_stage_partition_method,
            pipeline_use_boundary_markers=pipeline_use_boundary_markers,
            pipeline_boundary_start_marker=pipeline_boundary_start_marker,
            pipeline_boundary_end_marker=pipeline_boundary_end_marker,
        )

        # We manually set the communication op insertion rules
        global_configs.comm_op_insert_rules = {
            "all_reduce:tp:forward": {
                # "after_nodes": ['fc2', ],
                # "after_nodes": ['matmul', ],
                "after_markers": ['self_attention_linear_proj_weight', 'mlp_linear_fc2_weight', ],
                "after_nodes": ['contiguous', ],
                "interval": 1,
                "pass": 'forward',
                "parallelism": 'tp',
            },
            "all_reduce:dp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad_reduce_sum', ],
                "interval": 1,
                "pass": 'backward',
                "parallelism": 'dp',
            },
            "all_reduce:tp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad_reduce_sum', ],
                "skip_first_n": 4,  # Addition 1 gemm before and after transformer layer loop
                "interval": 4,      # Each transformer layer's gemm: qkv, out proj, mlp-fc1, mlp-fc2
                "pass": 'backward',
                "parallelism": 'tp',
            },
        }

        # Init distributed
        initialize_virtual_distributed(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            init_cpu_process_group=True,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron
        megatron_config = TransformerConfig(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=max(1, global_configs.tensor_model_parallel_size),
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=True,
        )
        setattr(megatron_config, "max_sequence_length", seq_len)
        setattr(megatron_config, "vocab_size", vocab_size)
        setattr(megatron_config, "microbatch_size", global_bs // num_micro_batches)

        # Model
        _model_loader_class = ShadowModelLoader if global_configs.use_shadow_model_loader else ModelLoader
        kwargs = {"device_str": device} if not global_configs.use_shadow_model_loader else {"fake_device_str": device}
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "max_seq_len": seq_len,
                "vocab_size": vocab_size,
                "device": device,
                "dtype": dtype,
                "megatron_config": megatron_config,
            },
            **kwargs,
        )
        model = model_loader.load()
        
        # Inputs
        input_ids = torch.ones(global_bs // num_micro_batches, seq_len, dtype=data_dtype, device=device)
        position_ids = torch.arange(seq_len, dtype=data_dtype, device=device).unsqueeze(0).expand_as(input_ids)
        attention_mask = torch.ones(global_bs // num_micro_batches, 1, seq_len, seq_len, dtype=torch.bool, device=device)
        decoder_input = None
        labels = torch.ones(global_bs // num_micro_batches, seq_len, dtype=data_dtype, device=device)
        input_args = ()
        input_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "decoder_input": decoder_input,
            "labels": labels,
        }

        # Profile
        profiler = CriusDistProfiler(
            profiled_data_path='./crius/offline/prof_data.pkl',
        )
        e2e_latency = profiler.profile(
            model=model,
            parallelism_plan=parallelism_plan,
            only_forward=global_configs.only_forward,
            base_input_args=input_args,
            base_input_kwargs=input_kwargs,
            partition_pipeline_stages_func=CriusParallelPlanner.partition_pipeline_stages,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")


if __name__ == '__main__':
    unittest.main()
