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
from transformers import AutoConfig, AutoModelForCausalLM
from crius.loader import ShadowModelLoader, ModelLoader

from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan
from crius.model.tf_gpt_model import GPTConfig, GPTModel
from crius.compile.graph.transform import transform_to_ir_graph
from crius.compile.graph.analysize import get_op_layer_mapping
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from crius.model.model_config import model_config



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
    def test_torch_model_profiling(self):
        """ Profile a naive torch model. """
        
        global_bs = 4
        seq_len = 2048
        num_micro_batches = 1
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = "cuda"
        enable_torch_profiler = True

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
        for _ in range(10):
            with torch.no_grad():
                _ = model(data[0])
        torch.cuda.synchronize(device)

        if not enable_torch_profiler:     
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            # Measure
            start_event.record()
            for _ in range(100):
                with torch.no_grad():
                    _ = model(data[0])
            end_event.record()    

            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / 100     
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
            
            print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")
        
        else:

            # 创建 SummaryWriter，指向一个日志目录
            writer = SummaryWriter('/app/tensorboard_profiler_logs')

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=tensorboard_trace_handler(writer.log_dir),  # ✅ 关键：使用 tensorboard_trace_handler
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            ) as prof:
                for _ in range(10):  # 运行多个 step 以触发 schedule
                    with torch.no_grad():
                        _ = model(data[0])
                    prof.step()  # ✅ 必须调用 prof.step() 来推进 profiler 状态

            # with torch.profiler.profile(
            #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ],
            #     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            #     # on_trace_ready=lambda x: x.export_chrome_trace('/app/torch_exec_trace.json'),
            #     record_shapes=True,
            #     profile_memory=True,
            #     with_stack=True,
            # ) as prof:
            #     for _ in range(1):
            #         with torch.no_grad():
            #             _ = model(input_ids)
            #     prof.step()
        
            # prof.export_chrome_trace('/app/torch_exec_trace.json')


    @unittest.skip("Pass.")
    def test_torch_gpt_model_profiling(self):
        global_bs = 16
        micro_bs = 4
        seq_len = 4096
        num_micro_batches = global_bs // micro_bs
        vocab_size = 50257
        hidden_size = 8192
        num_layers = 1
        num_heads = 1
        dtype = torch.float16  
        device = torch.device("cuda") 
        enable_torch_profiler = False

        def model_provider(**kwargs) -> nn.Module:
            config = GPTConfig(
                    vocab_size = vocab_size,
                    n_positions = seq_len,
                    n_embd = hidden_size,
                    n_layer = num_layers,
                    n_head = num_heads,
                    device = device,
                    dtype = dtype,
            )
            return GPTModel(config)

        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
                "dtype": dtype,
            },
            fake_device_str=device,
        )
        model = model_loader.load()

        # Inputs
        input_ids = torch.randint(0, vocab_size, (micro_bs, seq_len), dtype=torch.long).to(device)
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
            num_microbatches=num_micro_batches,
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
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)
        torch.cuda.synchronize(device)

        if not enable_torch_profiler:     
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            # Measure
            step = 10
            start_event.record()
            for _ in range(step):
                with torch.no_grad():
                    _ = model(input_ids)
            end_event.record()    

            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / step   
            print(f"========> Average latency: {avg_latency_ms} ms")  
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
            
            print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")
        
        else:

            # 创建 SummaryWriter，指向一个日志目录
           writer = SummaryWriter('/app/tensorboard_profiler_logs')

           with profile(
               activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
               schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
               on_trace_ready=tensorboard_trace_handler(writer.log_dir),  # ✅ 关键：使用 tensorboard_trace_handler
               record_shapes=True,
               profile_memory=True,
               with_stack=True,
               with_flops=True,
               with_modules=True,
           ) as prof:
               for _ in range(10):  # 运行多个 step 以触发 schedule
                   with torch.no_grad():
                       _ = model(input_ids)
                   torch.cuda.synchronize()
                   prof.step()  # ✅ 必须调用 prof.step() 来推进 profiler 状态

        
    # @unittest.skip("Pass.")
    def test_hf_gpt_model_profiling(self):
        micro_bs = 1
        global_bs = 1
        dtype = torch.float32  
        device = torch.device("cuda:0") 
        enable_torch_profiler = False
        profiler_layer_num = 1

        '''
        hugging-face model url: https://huggingface.co/openai-community/gpt2-xl
        '''

        args = argparse.Namespace(
            model="/build/Qwen3-30B-A3B",  
            # model="/build/gpt2-xl",  
            device=device,
            dtype=dtype,

        )

        def hf_model_provider(**kwargs) -> nn.Module:
            model_path = kwargs["model_path"]

            # 判断是否在fake tensor模式下
            with torch.no_grad():
                dummy = torch.empty(1)
                is_fake = isinstance(dummy, torch._subclasses.FakeTensor)

            config = AutoConfig.from_pretrained(model_path)
            config.use_cache = False

            config.num_hidden_layers = profiler_layer_num
            if hasattr(config, "n_layer"):
                config.n_layer = profiler_layer_num

            if is_fake:
                print(f"[Fake Mode] Creating model structure only: {model_path}")
                model = AutoModelForCausalLM.from_config(config)   
            else:
                print(f"[Real Mode] Loading pretrained weights: {model_path}")       
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    config=config,
                )

                # GPT-2 bloom etc.
                if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    if len(model.transformer.h) > profiler_layer_num:
                        model.transformer.h = nn.ModuleList([model.transformer.h[:profiler_layer_num]])
                # LLaMA  Qwen Mistral etc.
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    if len(model.model.layers) > profiler_layer_num:
                        model.model.layers = nn.ModuleList([model.model.layers[:profiler_layer_num]])
                else:
                    raise ValueError("Unsupported model type")

                model.config.use_cache = False 

            return model
 
 
        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model, "dtype": args.dtype},
            fake_device_str=args.device
        )
        model = model_loader.load()

        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)

        model_path = args.model.lower()
        import re
        # 定义关键词规则
        if re.search(r'gpt(?:[0-9\-_])*[0-9]', model_path) or \
            'gpt2' in model_path or 'gpt-neo' in model_path or 'gpt-j' in model_path or 'gpt-large' in model_path:
            model_config.num_layers = config.num_hidden_layers
            model_config.num_heads = config.num_attention_heads
            model_config.hidden_size = config.n_embd
            model_config.seq_len = config.n_ctx
            model_config.vocab_size = config.vocab_size

        if 'qwen' in model_path or \
            re.search(r'q?wen[0-9\-_]*[0-9]', model_path) or \
            'llama' in model_path :  # 匹配 qwen, qwen3, wen3 等
            model_config.num_layers = config.num_hidden_layers
            model_config.num_heads = config.num_attention_heads
            model_config.hidden_size = config.hidden_size
            model_config.seq_len = config.max_position_embeddings
            model_config.vocab_size = config.vocab_size
            if hasattr(config, "num_experts_per_tok"):
                model_config.if_moe = True
                model_config.top_k = config.num_experts_per_tok
            else:
                model_config.if_moe = False

        print(f"Model config:  num_layers = {model_config.num_layers}, \
              num_heads = {model_config.num_heads}, seq_len = {model_config.seq_len}, \
                vocab_size = {model_config.vocab_size}")

        input_ids = torch.randint(0, model_config.vocab_size, (micro_bs, model_config.seq_len), dtype=torch.long).to(device)
        graph = transform_to_ir_graph(model, input_ids)

        num_stages = 1
        base_layers = model_config.num_layers // num_stages
        start = 0
        layer_to_stage_mapping = []
        for i in range(num_stages):
            end = start + base_layers * (i+1)
            layer_to_stage_mapping.append(list(range(start, end)))
            start = end
        logging.debug(f"layer_to_stage_mapping = {layer_to_stage_mapping}")
        op_to_layer_mapping = get_op_layer_mapping(graph)
        num_micro_batches = global_bs // micro_bs

        parallelismPlan = ParallelismPlan(
            pipeline_type="1f1b",
            num_microbatches=num_micro_batches,
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
            only_forward=True,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms <=======")

        ################# Real profiling #################
        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model, "device": args.device, "dtype": args.dtype},
            device_str=device,
        )
        model = model_loader.load()


        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)
        torch.cuda.synchronize(device)

        if not enable_torch_profiler:     
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            # Measure
            step = 10
            start_event.record()
            for _ in range(step):
                with torch.no_grad():
                    _ = model(input_ids)
            end_event.record()    

            end_event.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / step   
            print(f"========> Average latency: {avg_latency_ms} ms")  
            stage_latencies_ms = [avg_latency_ms]
            measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
            
            print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")
            accu = 0.0
            accu = e2e_latency / measured_e2e_latency
            print(f"profiling latency = {e2e_latency:.3f}")
            print(f"accu = {accu:.3f}")
       

        else:

            # 创建 SummaryWriter，指向一个日志目录
            writer = SummaryWriter('/app/tensorboard_profiler_logs')

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=tensorboard_trace_handler(writer.log_dir),  # ✅ 关键：使用 tensorboard_trace_handler
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            ) as prof:
                for _ in range(10):  # 运行多个 step 以触发 schedule
                    with torch.no_grad():
                        _ = model(input_ids)
                    torch.cuda.synchronize()
                    prof.step()  # ✅ 必须调用 prof.step() 来推进 profiler 状态

           
if __name__ == '__main__':
    unittest.main()
