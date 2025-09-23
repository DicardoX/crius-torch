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
from crius.util import get_filename_include_path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim


def setup(rank, world_size):
    if not dist.is_available():
        return
    
    dist.init_process_group(
        backend = 'nccl',
        init_method = 'env://',
        world_size = world_size,
        rank = rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


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
    def test_torch_model_profiling(self):
        """ Profile a naive torch model. """
        
        # step 1: 初始化
        global_bs = 4
        seq_len = 2048
        num_micro_batches = 1
        hidden_size = 4096
        num_layers = 4
        dtype = torch.float16
        device = "cuda"

        micro_bs = global_bs // num_micro_batches
        
        # Inputs
        data = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]
        labels = [
            torch.randn(global_bs // num_micro_batches, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]

        # step 2： single card profiler
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
            profiled_data_path = get_filename_include_path(),
        )
        e2e_latency = profiler.profile(
            model=model,
            base_input_args=(data[0], ),
            parallelism_plan=parallelism_plan,
            only_forward=False,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        # step 3: single card training
        rank = int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
        world_size = int(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

        # 主进程打印信息
        if rank == 0:
            print(f"Starting profiling test on {world_size} GPU(s)")
            print(f"Global BS: {global_bs}, Micro BS: {micro_bs}, Seq Len: {seq_len}, Layers: {num_layers}")

        setup(rank, world_size)

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

        model.to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device = rank)

        # -------------------------------
        # 优化器
        # -------------------------------
        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

        def train_step(input_tensor, target_tensor):
            output = ddp_model(input_tensor)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, target_tensor)
            loss.backward()
            return loss


        # -------------------------------
        # 使用 PyTorch Profiler
        # -------------------------------
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/trace_rank_{rank}".format(rank=rank)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,  # 可视化算子来源
            with_flops=True,
        ) as prof:
            ddp_model.train()
            step_times = []
            for step in range(6):  # 对应 warmup=2, active=3
                start_event = torch.cuda.Event(enable_timing = True)
                end_event = torch.cuda.Event(enable_timing = True)
                optimizer.zero_grad(set_to_none=True)

                start_event.record()

                total_loss = 0.0
                for input_tensor, target_tensor in zip(data, labels):
                    loss = train_step(input_tensor, target_tensor)
                    total_loss += loss.item()

                optimizer.step()

                end_event.record()
                torch.cuda.synchronize()

                step_time_ms = start_event.elapsed_time(end_event)
                step_times.append(step_time_ms)

                # 记录 profiler step
                prof.step()

                if rank == 0:
                    if step < 2:
                        print(f"[Warmup] Step {step}, Loss: {total_loss:.4f}, Time: {step_time_ms:.2f} ms")
                    else:
                        print(f"[Active] Step {step}, Loss: {total_loss:.4f}, Time: {step_time_ms:.2f} ms")

            # 可选：输出平均迭代时间（仅 Active 阶段）
            if rank == 0 and len(step_times) > 2:
                avg_time_ms = sum(step_times[2:]) / len(step_times[2:])  # 排除 warmup
                print(f"\n✅ Average training step time (excl. warmup): {avg_time_ms:.2f} ms per step")
                print(f"✅ Estimated throughput: {global_bs / (avg_time_ms / 1000.0):.2f} samples/sec")


    def tearDown(self):
        """每个 test 后清理"""
        cleanup()


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


        # step 2： single card profiler

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
        labels = torch.randint(0, vocab_size, (micro_bs, seq_len), dtype=torch.long).to(device)
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
            profiled_data_path = get_filename_include_path(),
        )
        e2e_latency = profiler.profile(
            model=model,
            base_input_args=(input_ids, ),
            parallelism_plan=parallelismPlan,
            only_forward=False,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

        ################# Real profiling #################
        # step 3: single card training
        rank = int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
        world_size = int(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

        # 主进程打印信息
        if rank == 0:
            print(f"Starting profiling test on {world_size} GPU(s)")
            print(f"Global BS: {global_bs}, Micro BS: {micro_bs}, Seq Len: {seq_len}, Layers: {num_layers}")

        setup(rank, world_size)
        
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

        model.to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device = rank)

        # -------------------------------
        # 优化器
        # -------------------------------
        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

        def train_step(input_tensor, target_tensor):
            output = ddp_model(input_tensor)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, target_tensor)
            loss.backward()
            return loss
        
        # -------------------------------
        # 使用 PyTorch Profiler
        # -------------------------------
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/trace_rank_{rank}".format(rank=rank)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,  # 可视化算子来源
            with_flops=True,
        ) as prof:
            ddp_model.train()
            step_times = []
            for step in range(6):  # 对应 warmup=2, active=3
                start_event = torch.cuda.Event(enable_timing = True)
                end_event = torch.cuda.Event(enable_timing = True)
                optimizer.zero_grad(set_to_none=True)

                start_event.record()

                total_loss = 0.0
                for input_tensor, target_tensor in zip(input_ids, labels):
                    loss = train_step(input_tensor, target_tensor)
                    total_loss += loss.item()

                optimizer.step()

                end_event.record()
                torch.cuda.synchronize()

                step_time_ms = start_event.elapsed_time(end_event)
                step_times.append(step_time_ms)

                # 记录 profiler step
                prof.step()

                if rank == 0:
                    if step < 2:
                        print(f"[Warmup] Step {step}, Loss: {total_loss:.4f}, Time: {step_time_ms:.2f} ms")
                    else:
                        print(f"[Active] Step {step}, Loss: {total_loss:.4f}, Time: {step_time_ms:.2f} ms")

            # 可选：输出平均迭代时间（仅 Active 阶段）
            if rank == 0 and len(step_times) > 2:
                avg_time_ms = sum(step_times[2:]) / len(step_times[2:])  # 排除 warmup
                print(f"\n✅ Average training step time (excl. warmup): {avg_time_ms:.2f} ms per step")
                print(f"✅ Estimated throughput: {global_bs / (avg_time_ms / 1000.0):.2f} samples/sec")


    # @unittest.skip("Pass.")
    def test_hf_model_profiling(self):

        '''
        hugging-face model url: https://huggingface.co/openai-community/gpt2-xl
        '''

        def get_hf_model_config(model: str, trust_remote_code=True):

            config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)

            if hasattr(config, "torch_dtype"):
                model_config.dtype = config.torch_dtype

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

            print(f"Model config:  dtype = {model_config.dtype}, num_layers = {model_config.num_layers}, " +
                    f"num_heads = {model_config.num_heads}, seq_len = {model_config.seq_len}, " +
                    f"vocab_size = {model_config.vocab_size}") 
            

        def hf_model_provider(**kwargs) -> nn.Module:
            model_path = kwargs["model_path"]

            # 判断是否在fake tensor模式下
            with torch.no_grad():
                dummy = torch.empty(1)
                is_fake = isinstance(dummy, torch._subclasses.FakeTensor)

            config = AutoConfig.from_pretrained(model_path)
            config.use_cache = False

            config.num_hidden_layers = args.profiler_layer_num
            if hasattr(config, "n_layer"):
                config.n_layer = args.profiler_layer_num

            if is_fake:
                print(f"[Fake Mode] Creating model structure only: {model_path}")
                model = AutoModelForCausalLM.from_config(config)   
            else:
                print(f"[Real Mode] Loading pretrained weights: {model_path}")       
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=model_config.dtype,
                    low_cpu_mem_usage=True,
                    config=config,
                )

                # GPT-2 bloom etc.
                if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    if len(model.transformer.h) > args.profiler_layer_num:
                        model.transformer.h = nn.ModuleList([model.transformer.h[:args.profiler_layer_num]])
                # LLaMA  Qwen Mistral etc.
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    if len(model.model.layers) > args.profiler_layer_num:
                        model.model.layers = nn.ModuleList([model.model.layers[:args.profiler_layer_num]])
                else:
                    raise ValueError("Unsupported model type")

                model.config.use_cache = False 

            return model
 
        args = argparse.Namespace(
            # model="/build/Qwen3-30B-A3B",  
            # model="/build/Llama-2-7b-hf",  
            # model="/build/Llama-3.1-8B",  
            # model="/build/Qwen3-0.6B",  
            model="/build/gpt2-xl",  
            device=torch.device("cuda") ,
            micro_bs = 1,
            global_bs = 1,
            enable_torch_profiler = False,
            profiler_layer_num = 1,
            skip_profiler = False,
        )

        get_hf_model_config(args.model)

        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model},
            fake_device_str=args.device
        )
        model = model_loader.load()

        input_ids = torch.randint(0, model_config.vocab_size, (args.micro_bs, model_config.seq_len), dtype=torch.long).to(args.device)
        labels = torch.randint(0, model_config.vocab_size, (args.micro_bs, model_config.seq_len), dtype=torch.long).to(args.device) 
        if args.skip_profiler == False:
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
            num_micro_batches = args.global_bs // args.micro_bs

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
                profiled_data_path = get_filename_include_path(),
            )
            e2e_latency = profiler.profile(
                model=model,
                base_input_args=(input_ids, ),
                parallelism_plan=parallelismPlan,
                only_forward=False,
            )

            print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms <=======")

        ################# Real profiling #################
        
        rank = int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
        world_size = int(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

        # 主进程打印信息
        if rank == 0:
            print(f"Starting profiling test on {world_size} GPU(s)")
            print(f"Global BS: {args.global_bs}, Micro BS: {args.micro_bs}, Seq Len: {model_config.seq_len}, Layers: {model_config.num_layers}")

        setup(rank, world_size)

        _model_loader_class = ModelLoader
        model_loader = _model_loader_class(
            model_provider=hf_model_provider,
            model_kwargs={"model_path": args.model, "device": args.device, "dtype": model_config.dtype},
            device_str=args.device,
        )
        model = model_loader.load()

        model.to(args.device)
        ddp_model = DDP(model, device_ids=[rank], output_device = rank)

        # -------------------------------
        # 优化器
        # -------------------------------
        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

        def train_step(input_tensor, labels):
            outputs = ddp_model(input_tensor)
            logits = outputs.logits

            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :]  # [B, S-1, V]
            shift_labels = labels[:, 1:]      # [B, S-1]
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
            return loss


        # -------------------------------
        # 使用 PyTorch Profiler
        # -------------------------------
        
        if not args.enable_torch_profiler:     
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)  
            ddp_model.train()

            # Warmup
            for _ in range(10):
                _ = train_step(input_ids, labels)
            torch.cuda.synchronize(args.device)

            # Measure
            step = 100
            start_event.record()
            for _ in range(step):
                _ = train_step(input_ids, labels)
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
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/trace_rank_{rank}".format(rank=rank)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,  # 可视化算子来源
                with_flops=True,
            ) as prof:
                ddp_model.train()
                step_times = []
                for step in range(6):  # 对应 warmup=2, active=3
                    start_event = torch.cuda.Event(enable_timing = True)
                    end_event = torch.cuda.Event(enable_timing = True)
                    optimizer.zero_grad(set_to_none=True)

                    start_event.record()

                    loss = train_step(input_ids, labels)

                    optimizer.step()

                    end_event.record()
                    torch.cuda.synchronize()

                    step_time_ms = start_event.elapsed_time(end_event)
                    step_times.append(step_time_ms)

                    # 记录 profiler step
                    prof.step()

                    if rank == 0:
                        if step < 2:
                            print(f"[Warmup] Step {step}, Loss: {loss:.4f}, Time: {step_time_ms:.2f} ms")
                        else:
                            print(f"[Active] Step {step}, Loss: {loss:.4f}, Time: {step_time_ms:.2f} ms")

                # 可选：输出平均迭代时间（仅 Active 阶段）
                if rank == 0 and len(step_times) > 2:
                    avg_time_ms = sum(step_times[2:]) / len(step_times[2:])  # 排除 warmup
                    print(f"\n✅ Average training step time (excl. warmup): {avg_time_ms:.2f} ms per step")

           
if __name__ == '__main__':
    unittest.main()
