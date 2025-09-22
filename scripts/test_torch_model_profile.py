#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(current_dir)
if app_root not in sys.path:
    sys.path.insert(0, app_root)
    
from typing import Dict, Set
import unittest
import logging
from functools import partial
from collections import OrderedDict
import time
import numpy as np
from megatron.core import parallel_state
import torch
from torch import nn

from crius.loader import (ModelLoader, ShadowModelLoader)
from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan

from megatron.core.tensor_parallel.layers import (ColumnParallelLinear,RowParallelLinear)
from megatron.core.model_parallel_config import ModelParallelConfig



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

        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.module import MegatronModule

        from crius.train_backend.megatron.utils import initialize_distributed

        def init_method_normal(tensor):
            nn.init.normal_(tensor, mean=0.0, std=0.02)

        class MegatronToyLayer(MegatronModule):
            # TODO(chunyu): Replace nn.Linear with Megatron's tensor parallelized linear layer.
            
            def  __init__(self, hidden_size: int, config: TransformerConfig, device: str, dtype: str = torch.float16) -> None:
                super().__init__(config=config)
                # self.fc1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
                # self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)

                self.fc1 = ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=hidden_size * 4,
                    config=config,
                    init_method=init_method_normal,
                    gather_output=False,
                    bias=True,
                    skip_bias_add=False,
                    is_expert=False,
                    # tp_group=None,
                )

                self.fc2 = RowParallelLinear(
                    input_size=hidden_size * 4,
                    output_size=hidden_size,
                    config=config,
                    init_method=init_method_normal,
                    input_is_parallel=False,
                    bias=True,
                    skip_bias_add=False,
                    is_expert=False,
                    # tp_group=None,
                )

                self.activation = nn.GELU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = self.activation(x)
                x = self.dropout(x)

                x = self.fc2(x)

                return x
                # x = self.fc1(x)
                # _x = torch.
                # return self.fc2(x)

        class MegatronToyModel(MegatronModule):
            """ Toy Megatron model. """

            def __init__(self, hidden_size: int, num_layers: int, config: ModelParallelConfig, device: str, dtype: str = torch.float16) -> None:
                super().__init__(config=config)
                self.device = device
                self.num_layers = num_layers
                self.layers = nn.ModuleList([MegatronToyLayer(hidden_size, config, device, dtype) for _ in range(num_layers)])

                self.output_layer = ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    config=config,
                    init_method=init_method_normal,
                    gather_output=True,
                    bias=True,
                    skip_bias_add=False,
                    is_expert=False,
                    #tp_group=None,
                )

                self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)

            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                hidden_states = input_
                for i in range(self.num_layers):
                    hidden_states = self.layers[i](hidden_states)

                hidden_states = self.output_layer(hidden_states)
                return self.layer_norm(hidden_states)

        def create_transformer_config(hidden_size=1024):
            config = TransformerConfig(
                hidden_size=hidden_size,
                ffn_hidden_size=hidden_size * 4,
                num_layers=1,
                num_attention_heads=16,
                add_bias_linear=True,
                apply_residual_connection_post_layernorm=False,
                fp16=False,
                bf16=False,
                use_cpu_initialization=False,
                perform_initialization=True,
                params_dtype=torch.float32,
                sequence_parallel=False,
                gradient_accumulation_fusion=False,
            )

            def init_method_normal(tensor):
                nn.init.normal_(tensor, mean=0.0, std=0.02)

            config.init_method = init_method_normal
            return config

        def _model_provider(hidden_size: int, num_layers: int,config: ModelParallelConfig, device: str, dtype: str) -> nn.Module:
            """ Model provider function. """
            #config = create_transformer_config(hidden_size)
            return MegatronToyModel(hidden_size, num_layers, config, device, dtype=dtype)

        def initialize_distributed_and_tp(tensor_parallel_size: int = 1):
            """Initialize distributed training and tensor parallelism."""
    
            # Initialize PyTorch distributed
            if not torch.distributed.is_initialized():
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
        
                torch.cuda.set_device(local_rank)
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=int(os.environ.get('RANK', 0))
                )
    
            # Initialize Megatron model parallel
            if not parallel_state.model_parallel_is_initialized():
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=tensor_parallel_size,
                    pipeline_model_parallel_size=1,
                )
    
            # Initialize CUDA RNG tracker for Megatron
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
            model_parallel_cuda_manual_seed(1234)
    
            return parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()

        tensor_parallel_size = int(os.environ.get('WORLD_SIZE', 2))  # Default to 2 for TP
        tp_rank, tp_world_size = initialize_distributed_and_tp(tensor_parallel_size)

        global_bs = 8
        seq_len = 2048
        num_micro_batches = 2
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = f"cuda:{torch.cuda.current_device()}"

        try:
            #torch.manual_seed(42)
            #torch.cuda.manual_seed_all(42)
            #torch.cuda.model_parallel_rng = torch.cuda.manual_seed(42)

            #initialize_distributed(
             #   tensor_model_parallel_size=1,
             #   pipeline_model_parallel_size=1,
                #expert_model_parallel_size=1
            #)
            print("Distributed initialization successful")
        except Exception as e:
            print(f"Distributed initialization failed: {e}")
            return

        # transformer_config = TransformerConfig()

        #
        try:
            # _model_loader_class = ShadowModelLoader
            # model_loader = _model_loader_class(
            #     model_provider=model_provider,
            #     model_kwargs={
            #         "hidden_size": hidden_size,
            #         "num_layers": num_layers,
            #         "device": device,
            #         "dtype": dtype,
            #     },
            #     # device_str=device,
            #     fake_device_str=device,
            # )
            # model = model_loader.load()
            
            #from deepspeed.runtime.activation_checkpointing.checkpointing import _CUDA_RNG_STATE_TRACKER, _MODEL_PARALLEL_RNG_TRACKER_NAME
            #_CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)
            
            config = ModelParallelConfig(
                tensor_model_parallel_size=tp_world_size,
                pipeline_model_parallel_size=1,
            )
            model = _model_provider(hidden_size, num_layers, config, device, dtype)
            print(f"Model created successfully on {device}")
        except Exception as e:
            print(f"Model created failed: {e}")
            return
        
        # Inputs
        try:
            torch.manual_seed(1234)
            data = [
                torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                    for _ in range(num_micro_batches)
                ]
            print(f"Input created successfully on {device}")
        except Exception as e:
            print(f"Input created failed: {e}")
            return

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
        #profiler = CriusDistProfiler(
        #    profiled_data_path='./crius/offline/prof_data.pkl',
        #)

        print(model)

        #exported_model = torch.jit.script(model)

        #e2e_latency = profiler.profile(
        #    model=exported_model,
        #    base_input_args=(data[0], ),
        #    parallelism_plan=parallelism_plan,
        #    only_forward=True,
        #)

        #print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        from crius.compile import (compile, CompileConfigs)

        compile_configs = CompileConfigs(
            (data[0],),
            parallelism_plan=parallelism_plan,
        )
        
        graph = compile(model, compile_configs)
        # print(graph)

        # for node in graph.nodes:
        #     if node.name in graph.node_layer_mapping:
        #         layer_info = graph.node_layer_mapping[node.name]
        #         #if layer_info['layer_class'] == 'Linear':
        #         print(f" {node.name:20} ({node.op:15}) belongs to {layer_info['layer_class']}-{layer_info['module_path']} layer")
        
    

if __name__ == '__main__':
    unittest.main()

