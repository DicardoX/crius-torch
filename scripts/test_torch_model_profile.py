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
import argparse
import gc
import unittest
import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from crius.loader import ShadowModelLoader, ModelLoader

from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan
from crius.model.tf_gpt_model import GPTConfig, GPTModel
from crius.compile.graph.transform import transform_to_ir_graph
from crius.compile.graph.analysize import get_op_layer_mapping


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return self.fc2(x)


class ToyModel(nn.Module):
    """ Toy PyTorch model. """

    def __init__(
            self, hidden_size: int, num_layers: int, device: str, dtype: torch.dtype = torch.float16
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


def model_provider(hidden_size: int, num_layers: int, device: str) -> nn.Module:
    """ Model provider function. """
    return ToyModel(hidden_size, num_layers, device)


class TorchModelProfileTest(unittest.TestCase):
    """ Basic torch model disaggregated profiling test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """
    @unittest.skip("Pass.")
    def test_torch_model_offline_profiling1(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=1,
            max_batch_size=4,
            batch_size_step="power-of-2",
            min_seq_len=32,
            max_seq_len=1024,
            seq_len_step="power-of-2",
            gemm_op_type=torch.mul,
            gemm_left_op_hidden_dims=[128, ],
            gemm_right_op_hidden_dims=[128, ],
            dtype=torch.float16,
            device="cuda:0",
            num_warmup_steps=5,
            num_profile_steps=10,
            force_overwrite=True,
        )
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        
        print("\ntorch.mul ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.tril
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch tril ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.eq
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch eq ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.add
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch add ===========================")
        offline_profiler.profile_single_input_gemm()

    @unittest.skip("Pass.")
    def test_torch_model_offline_profiling(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=1,
            max_batch_size=4,
            batch_size_step="power-of-2",
            min_seq_len=32,
            max_seq_len=1024,
            seq_len_step="power-of-2",
            gemm_left_op_hidden_dims=[128, ],
            gemm_right_op_hidden_dims=[128, ],
            dtype=torch.float16,
            device="cuda:1",
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
        seq_len = 128
        num_micro_batches = 2
        hidden_size = 128
        num_layers = 4
        dtype = torch.float16
        device = "cuda:0"

        # Model
        # _model_loader_class = ModelLoader
        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
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
            },
            device_str=device,
        )
        model = model_loader.load()

        # Warmup
        for _ in range(100):
            _ = model(data[0])
        torch.cuda.synchronize(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  

        # Measure
        start_event.record()
        for _ in range(1000):
            _ = model(data[0])
        end_event.record()    

        torch.cuda.synchronize(device)
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 1000     
        stage_latencies_ms = [avg_latency_ms]
        measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        
        print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")

    @unittest.skip("Pass.")
    def test_torch_gpt_model_profiling(self):
        seq_len = 1024
        num_micro_batches = 1
        global_bs = 4
        vocab_size = 50257
        num_layers = 12
        num_heads = 8
        hidden_size = 512
        dtype = torch.long  
        device = torch.device("cuda:0") 

        def model_provider(**kwargs) -> nn.Module:
            config = GPTConfig(
                    vocab_size = vocab_size,
                    n_positions = 1024,
                    n_layer = num_layers,
                    n_head = num_heads,
                    n_embd= hidden_size
            )
            return GPTModel(config)

        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "device": device,
            },
            fake_device_str=device,
        )
        model = model_loader.load()

        input_ids = torch.randint(0, vocab_size, (global_bs // num_micro_batches, seq_len), dtype=dtype).to(device)
        graph = transform_to_ir_graph(model, input_ids)

        num_stages = 1
        base_layers = num_layers // num_stages
        start = 0
        layer_to_stage_mapping = []
        for i in range(num_stages):
            end = start + base_layers * (i+1)
            layer_to_stage_mapping.append(list(range(start, end)))
            start = end
        logging.debug(f"layer_to_stage_mapping = {layer_to_stage_mapping}")
        op_to_layer_mapping = get_op_layer_mapping(graph)

        parallelismPlan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=1, 
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
            base_input_args=(input_ids, ),
            parallelism_plan=parallelismPlan,
        )

        measured_e2e_latency =np.max(e2e_latency) * (num_micro_batches - 1) + np.sum(e2e_latency)
        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms,   measured end-to-end latency: {(measured_e2e_latency):.3f} ms <=======")

        ################# Real profiling #################
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "device": device,
            },
            device_str=device,
        )
        model = model_loader.load()

        # Warmup
        for _ in range(100):
            _ = model(input_ids)
        torch.cuda.synchronize(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  

        # Measure
        start_event.record()
        for _ in range(1000):
            _ = model(input_ids)
        end_event.record()    

        torch.cuda.synchronize(device)
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 1000     
        stage_latencies_ms = [avg_latency_ms]
        measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        
        print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")
    @unittest.skip("Pass.")
    def test_hf_gpt_model_profiling(self):
        micro_bs = 8
        global_bs = 32
        dtype = torch.long  
        device = torch.device("cuda:0") 

        '''
        hugging-face model url: https://huggingface.co/openai-community/gpt2-xl
        '''

        args = argparse.Namespace(
            model="/workspace/gpt2-xl",  
            device="cuda:0"
        )

        def hf_model_provider(**kwargs) -> nn.Module:
            model_path = kwargs["model_path"]
            config = transformers.GPT2Config.from_pretrained(model_path)
            config.use_cache = False
            model = GPT2LMHeadModel(config)
            
            model.half()
            return model
        
        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model, "device": args.device},
            fake_device_str=args.device
        )
        model = model_loader.load()

        config = transformers.GPT2Config.from_pretrained(args.model)

        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        hidden_size = config.n_embd
        seq_len = config.n_ctx
        vocab_size = config.vocab_size
        print(f"Model config: num_layers = {num_layers}, num_heads = {num_heads}, seq_len = {seq_len},vocab_size = {vocab_size}")

        input_ids = torch.randint(0, vocab_size, (micro_bs, seq_len), dtype=dtype).to(device)
        graph = transform_to_ir_graph(model, input_ids)

        num_stages = 1
        base_layers = num_layers // num_stages
        start = 0
        layer_to_stage_mapping = []
        for i in range(num_stages):
            end = start + base_layers * (i+1)
            layer_to_stage_mapping.append(list(range(start, end)))
            start = end
        logging.debug(f"layer_to_stage_mapping = {layer_to_stage_mapping}")
        op_to_layer_mapping = get_op_layer_mapping(graph)

        parallelismPlan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=1, 
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
            base_input_args=(input_ids, ),
            parallelism_plan=parallelismPlan,
        )

        num_micro_batches = global_bs // micro_bs
        measured_e2e_latency =np.max(e2e_latency) * (num_micro_batches - 1) + np.sum(e2e_latency)
        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms,   measured end-to-end latency: {(measured_e2e_latency):.3f} ms <=======")

        ################# Real profiling #################
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model, "device": args.device},
            device_str=device,
        )
        model = model_loader.load()

        # Warmup
        for _ in range(100):
            _ = model(input_ids)
        torch.cuda.synchronize(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  

        # Measure
        start_event.record()
        for _ in range(1000):
            _ = model(input_ids)
        end_event.record()    

        torch.cuda.synchronize(device)
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 1000     
        stage_latencies_ms = [avg_latency_ms]
        measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        
        print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")

if __name__ == '__main__':
    unittest.main()
