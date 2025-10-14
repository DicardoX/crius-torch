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

from crius.loader import (ModelLoader, ShadowModelLoader)
from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan
from crius.global_envs import global_configs


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: str = torch.float16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return self.fc2(x)


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


def model_provider(hidden_size: int, num_layers: int, device: str, dtype: str) -> nn.Module:
    """ Model provider function. """
    return ToyModel(hidden_size, num_layers, device, dtype=dtype)


class TorchModelProfileTest(unittest.TestCase):
    """ Basic torch model disaggregated profiling test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    # @unittest.skip("Pass.")
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

    @unittest.skip("Pass.")
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
            hidden_size: int, num_layers: int, max_seq_len: int, vocab_size: int, device: str, dtype: str, 
            megatron_config: Any = None, use_toy: bool = False,
        ) -> nn.Module:
            """ Model provider function. """
            if use_toy:
                return MegatronToyModel(hidden_size, num_layers, device, dtype=dtype, megatron_config=megatron_config)
            else:
                return GPTModel(megatron_config, vocab_size, max_seq_len, device=device)


        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        data_dtype = torch.long
        dtype = torch.float16
        device = "cuda"
        vocab_size = 50000
        use_toy = False
        
        global_configs.training_backend = "megatron"
        global_configs.tensor_model_parallel_size = 1
        global_configs.pipeline_model_parallel_size = 1
        global_configs.data_parallel_size = 2
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
            # "all_reduce:tp": {
            #     # "after_nodes": ['fc2', ],
            #     # "after_nodes": ['matmul', ],
            #     "after_markers": ['self_attention_linear_proj_weight', 'mlp_linear_fc2_weight', ],
            #     "after_nodes": ['contiguous', ],
            #     "interval": 1,
            #     "pass": 'forward',
            #     "parallelism": 'tp',
            # },
            "all_reduce:dp": {
                "after_markers": [],
                "after_nodes": ['weight_grad_reduce_sum', ],
                "interval": 1,
                "pass": 'backward',
                "parallelism": 'dp',
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

        # Profile
        profiler = CriusDistProfiler(
            profiled_data_path='./crius/offline/prof_data.pkl',
        )
        e2e_latency = profiler.profile(
            model=model,
            base_input_args=input_args,
            base_input_kwargs=input_kwargs,
            parallelism_plan=parallelism_plan,
            only_forward=global_configs.only_forward,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        exit(0)

        ################# Real profiling #################
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
                "dtype": dtype,
                "megatron_config": megatron_config,
            },
            device_str=device,
        )
        model = model_loader.load()

        # Warmup
        for _ in range(100):
            with torch.no_grad():
                _ = model(data[0])
        torch.cuda.synchronize(device)
    
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


if __name__ == '__main__':
    unittest.main()
