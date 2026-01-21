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
from crius.core.grid_planner.planner import CriusParallelPlanner
from crius.model_parallel.utils import ParallelismPlan
from crius.global_envs import global_configs, logger


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
    
    When used in multi-node execution, use: 
        `torchrun --nnodes X --nproc-per-node N --node_rank R --master_addr "MASTER_IP" --master_port PORT script.py`
    
    NOTE: In docker container, the used ssh port cannot be the same as the host (i.e., 22), must be changed to 
          another port.
    
          If hang in `torch.distributed.barrier()`, maybe use `export NCCL_P2P_LEVEL=NVL` to disable non-nvlink 
          GPU P2P access, see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level
    """

    @unittest.skip("Pass.")
    def test_one_torch_operator(self):
        """ Profile one torch operator in a hackable manner. """

        # TODO(chunyu): Debug why there is a large gap between offline and megatron's online P2P overhead.

        num_warmup_steps = 1
        num_profile_steps = 1
        use_pytorch_profiler = False
        use_torch_empty_to_init_weight = True
        use_transposed_weight = True

        torchrun_world_size = int(os.environ['WORLD_SIZE'])
        # Torch setup for distributed training
        local_rank = int(os.environ['LOCAL_RANK'])
        device_id = torch.device(f'cuda:{local_rank}')
        global_rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            world_size=torchrun_world_size, 
            rank=global_rank, 
            device_id=torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}"),
            backend="nccl",
        )

        # comm_tensor = torch.empty((1024, 4, 4096), device=torch.cuda.current_device(), dtype=torch.float16)
        comm_tensor = torch.empty((1871831040,), device=torch.cuda.current_device(), dtype=torch.float16)
        # comm_tensor = torch.empty((1219321856,), device=torch.cuda.current_device(), dtype=torch.float16)
        torch.cuda.synchronize()        

        # # GEMM
        # left_op = torch.randn((1024, 4, 4096), device="cuda", dtype=torch.float16)
        # right_op = torch.empty((6144, 4096), device="cuda", dtype=torch.float16) \
        #     if use_torch_empty_to_init_weight else torch.randn((6144, 4096), device="cuda", dtype=torch.float16)
        # torch.cuda.synchronize()

        # # BADDBMM
        # left_op = torch.empty((64, 512, 512), device=torch.cuda.current_device(), dtype=torch.float16)
        # mid_op = torch.empty((64, 512, 128), device=torch.cuda.current_device(), dtype=torch.float16)
        # right_op = torch.empty((64, 128, 512), device=torch.cuda.current_device(), dtype=torch.float16)
        # # result_op = torch.empty((64, 512, 512), device=torch.cuda.current_device(), dtype=torch.float16)
        # torch.cuda.synchronize()

        # # Masked fill
        # input_op = torch.empty((4, 32, 1024, 1024), device="cuda", dtype=torch.float16)
        # input_op = input_op.float()  # Ensure fp32 input
        # mask = torch.ones((4, 1, 1024, 1024), device="cuda", dtype=torch.bool)
        # value = -10000.0
        # torch.cuda.synchronize()

        for _ in range(num_warmup_steps):
            if use_transposed_weight:
                # _ = torch.matmul(left_op, right_op.t())
                # input_op.masked_fill_(mask, value)
                # _ = torch.baddbmm(left_op, mid_op, right_op, beta=0.0, alpha=0.01,)
                torch.distributed.all_reduce(comm_tensor)
            else:
                # _ = torch.matmul(left_op, right_op)
                # input_op.masked_fill_(mask, value)
                # _ = torch.baddbmm(left_op, mid_op, right_op, beta=0.0, alpha=0.01,)
                torch.distributed.all_reduce(comm_tensor)
        torch.cuda.synchronize()
        
        if use_pytorch_profiler:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        # Measure
        start_event.record()
        for _ in range(num_profile_steps):
            if use_pytorch_profiler:
                prof.step()
            
            if use_transposed_weight:
                # _ = torch.matmul(left_op, right_op.t())
                # input_op.masked_fill_(mask, value)
                with torch.no_grad():
                    # _ = torch.baddbmm(left_op, mid_op, right_op, beta=0.0, alpha=0.01,)
                    torch.distributed.all_reduce(comm_tensor)
            else:
                # _ = torch.matmul(left_op, right_op)
                # input_op.masked_fill_(mask, value)
                with torch.no_grad():
                    # _ = torch.baddbmm(left_op, mid_op, right_op, beta=0.0, alpha=0.01,)
                    torch.distributed.all_reduce(comm_tensor)
        
        end_event.record()
        end_event.synchronize()

        if use_pytorch_profiler:
            prof.stop()
            prof.export_chrome_trace('/app/torch_op_trace.json')

        op_latency = start_event.elapsed_time(end_event) / num_profile_steps
        print(f"\n=======> Operator latency: {(op_latency):.3f} ms")

    @unittest.skip("Pass.")
    def test_torch_model_offline_profiling(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=2,
            max_batch_size=4,
            batch_size_step="power-of-2",
            # min_seq_len=1024,
            # max_seq_len=2048,
            min_seq_len=128,
            max_seq_len=2048,
            seq_len_step="power-of-2",
            dtype=torch.float16,
            device="cuda",
            num_warmup_steps=5,
            num_profile_steps=10,
            force_overwrite=True,
            use_torch_empty_to_init_weight=True,
            use_transposed_weight=True,
            use_megatron_attention_softmax_in_fp32=True,
            gemm_left_op_hidden_dims=[      # Only consider forward ops
                1024, 2048, 4096, 8192, 16384, 
            ],
            gemm_right_op_hidden_dims=[
                1024, 2048, 4096, 3072, 6144, 12288, 8192, 16384, 12500, 25000, 50000, 
            ],
            bmm_op_hidden_dims=[
                1024, 2048, 4096, 8192,
            ],
            baddbmm_op_hidden_dims=[
                1024, 2048, 4096, 8192,
            ],
            num_attn_heads_per_rank=[
                8, 16, 32,
            ],
            layernorm_op_hidden_dims=[
                1024, 2048, 4096, 8192,
            ],
            dropout_op_hidden_dims=[
                1024, 2048, 4096, 8192,
            ],
            dropout_op_num_attn_heads=[
                8, 16, 32,
            ],
            masked_fill_op_num_attn_heads=[
                8, 16, 32,
            ],
            softmax_op_num_attn_heads=[
                8, 16, 32,
            ],
            comm_data_sizes = [
                # 2**i for i in range(1,28)] + [
                # 2**i for i in range(1,31)] + [
                # Special volumes
                # 102400000, 204800000, 409600000, 50331648 // 2, 50331648, 100663296,
                # 6647250944,
                1871831040,
            ],
        )
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )

        # offline_profiler.profile(opcode="gemm")
        # offline_profiler.profile(opcode="bmm")
        # offline_profiler.profile(opcode="baddbmm")
        # offline_profiler.profile(opcode="layernorm")
        # offline_profiler.profile(opcode="dropout")
        # offline_profiler.profile(opcode="masked_fill_")
        # offline_profiler.profile(opcode="softmax")

        offline_profiler.profile(opcode="all_reduce")
        # offline_profiler.profile(opcode="p2p")

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

        global_bs = 16
        seq_len = 2048
        num_micro_batches = 8
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 32
        data_dtype = torch.long
        dtype = torch.float16
        device = "cpu"
        vocab_size = 50000

        global_configs.num_nodes = 2
        global_configs.num_devices_per_node = 4

        global_configs.tensor_model_parallel_size = 4
        global_configs.pipeline_model_parallel_size = 2
        global_configs.data_parallel_size = 1
        global_configs.training_backend = "megatron"
        global_configs.only_forward = False
        # global_configs.enable_backward_graph_generation = True
        global_configs.enable_backward_graph_generation = False
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
        # pipeline_boundary_end_marker = ['dropout', 'add', ]
        # pipeline_boundary_start_marker = ['input_layernorm', ]
        pipeline_boundary_end_marker = ['final_layernorm', ]
        # pipeline_forced_p2p_comm_overhead_ms = 10.0
        pipeline_forced_p2p_comm_overhead_ms = -1.0
        forced_grad_sync_latency_scale_factor = 4.0     # Manually tuned
        forced_p2p_latency_scale_factor = 3.0           # Manually tuned

        parallelism_plan = ParallelismPlan(
            pipeline_type="1f1b",
            pipeline_parallel_degree=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            num_attn_heads=num_attention_heads,
            num_microbatches=num_micro_batches,
            tensor_parallel_degree=global_configs.tensor_model_parallel_size,
            data_parallel_degree=global_configs.data_parallel_size,
            forced_grad_sync_latency_scale_factor=forced_grad_sync_latency_scale_factor,
            forced_p2p_latency_scale_factor=forced_p2p_latency_scale_factor,
            pipeline_stage_partition_method=pipeline_stage_partition_method,
            pipeline_overlap_p2p_comm=False,
            pipeline_forced_p2p_comm_overhead_ms=pipeline_forced_p2p_comm_overhead_ms,
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
                "after_nodes": ['autograd_apply', ],
                "interval": 1,
                "pass": 'forward',
                "parallelism": 'tp',
            },
            "all_reduce:dp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad', ],
                "interval": 1,
                "pass": 'backward',
                "parallelism": 'dp',
            },
            "all_reduce:tp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad', ],
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
            data_parallel_size=global_configs.data_parallel_size,
            init_cpu_process_group=True,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron
        megatron_config = TransformerConfig(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=True,
        )
        setattr(megatron_config, "device_free_profile", True)
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
            check_oom=False,
        )

        print(f"\n=======> Estimated end-to-end latency: {(e2e_latency):.3f} ms")

    @unittest.skip("Pass.")
    def test_meagtron_ground_truth_gpt(self):
        """ Run the parallelized Megatron model as the ground truth, using Megatron's parallelization codes. """

        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import mpu, tensor_parallel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config, get_attr_wrapped_model

        from crius.train_backend.megatron.utils import initialize_distributed
        from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
        from crius.train_backend.megatron.transformer_layer import TransformerLayer
        from crius.train_backend.megatron.gpt_model import GPTModel
        from crius.train_backend.megatron.training.trainer import Trainer, ModelType
        from crius.train_backend.megatron.training.utils import (setup_model, get_train_data_iterator, 
                                                                 get_optimizer_and_scheduler, get_input_batch)

        def model_provider(
            hidden_size: int, num_layers: int, max_seq_len: int, vocab_size: int, device: str, dtype: str, 
            megatron_config: Any = None, use_toy: bool = False,
        ) -> nn.Module:
            """ Model provider function. """
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            return GPTModel(megatron_config, vocab_size, max_seq_len, pre_process, post_process, device=device)

        model_type = ModelType.GPT
        global_bs = 16
        seq_len = 1024
        num_micro_batches = 4
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 32
        dtype = torch.float16
        device = "cuda"
        vocab_size = 50000
        num_iters = 5
        
        global_configs.training_backend = "megatron"
        global_configs.tensor_model_parallel_size = 4
        global_configs.pipeline_model_parallel_size = 2
        global_configs.data_parallel_size = 1
        global_configs.only_forward = False

        # Get number of microbatches
        # In Megatron, each DP replica constantly processes `micro_batch_size` samples per micro-batch, 
        # only changing the number of micro-batches to maintain constant global batch size.
        micro_batch_size =  global_bs // num_micro_batches
        assert global_bs % (micro_batch_size * global_configs.data_parallel_size) == 0, \
            f"Global batch size {global_bs} must be divisible by micro-batch-size {micro_batch_size} " + \
            f"x data parallel size {global_configs.data_parallel_size}."
        num_micro_batches = global_bs // (micro_batch_size * global_configs.data_parallel_size)
        global_bs = micro_batch_size * num_micro_batches    # of this DP rank

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
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=(dtype == torch.float16),
        )
        # NOTE: We monkey patch all input configuratons into `megatron_config`, 
        # instead of explicitly pass-in into the arguments of `Trainer()`.
        setattr(megatron_config, "global_batch_size", global_bs)
        setattr(megatron_config, "num_micro_batches", num_micro_batches)
        setattr(megatron_config, "sequence_length", seq_len)
        setattr(megatron_config, "num_iters", num_iters)
        setattr(megatron_config, "dtype", dtype)

        # Model
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_seq_len": seq_len,
            "vocab_size": vocab_size,
            "device": device,
            "dtype": dtype,
            "megatron_config": megatron_config,
        }
        model = model_provider(**model_kwargs) if model_provider else None
        model = setup_model(model)

        train_data_iterator = get_train_data_iterator(
            vocab_size, global_bs, num_micro_batches, seq_len, num_iters=num_iters,
        )
        (optimizer, opt_param_scheduler) = get_optimizer_and_scheduler(
            model, global_bs, num_iters, fp16=(dtype == torch.float16), params_dtype=dtype, 
        )

        # Definition of forward step function
        def __forward_step_func(data_iterator, model):
            """ Vanilla forward step function. """
   
            def __loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
                """ Loss function definition. """
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                # If you have data parallel reduce loss across data parallel groups.
                # If pipeline parallel, loss computation is done only in last stage.
                return loss, {"lm loss": loss}

            megatron_config = get_attr_wrapped_model(model, attr="config")
            # Input batch
            (tokens, labels, loss_mask, 
             attention_mask, position_ids) = get_input_batch(
                 data_iterator=data_iterator,
                 micro_batch_size=(
                     megatron_config.global_batch_size // megatron_config.num_micro_batches
                 ),
                 seq_len=megatron_config.sequence_length,
                 pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            )
            # Forward
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor,  partial(__loss_func, loss_mask)

        # Trainer
        trainer = Trainer(
            megatron_transformer_config=megatron_config,
            forward_step_func=__forward_step_func,
            model=model,
            model_type=model_type,
            train_data_iterator=train_data_iterator,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            pipeline_strategy="1f1b",
            forward_only=False,
            use_pytorch_profiler=True,
        )
        trainer.train()

    @unittest.skip("Pass.")
    def test_meagtron_ground_truth_moe(self):
        """ Run the parallelized Megatron model as the ground truth, using Megatron's parallelization codes. """

        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import mpu, tensor_parallel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config, get_attr_wrapped_model

        from crius.train_backend.megatron.utils import initialize_distributed
        from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
        from crius.train_backend.megatron.transformer_layer import TransformerLayer
        from crius.train_backend.megatron.qwen_moe_model import QWenMoEModel
        from crius.train_backend.megatron.training.trainer import Trainer, ModelType
        from crius.train_backend.megatron.training.utils import (setup_model, get_train_data_iterator, 
                                                                 get_optimizer_and_scheduler, get_input_batch)

        def model_provider(
            hidden_size: int, num_layers: int, max_seq_len: int, vocab_size: int, device: str, dtype: str, 
            megatron_config: Any = None, use_toy: bool = False,
        ) -> nn.Module:
            """ Model provider function. """
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            return QWenMoEModel(megatron_config, vocab_size, max_seq_len, pre_process, post_process, device=device)

        model_type = ModelType.MoE
        global_bs = 8
        seq_len = 1024
        num_micro_batches = 2
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 4
        dtype = torch.float16
        device = "cuda"
        vocab_size = 50000
        num_iters = 5
        
        global_configs.training_backend = "megatron"
        global_configs.tensor_model_parallel_size = 2
        global_configs.pipeline_model_parallel_size = 1
        global_configs.data_parallel_size = 1
        global_configs.only_forward = False

        # Get number of microbatches
        # In Megatron, each DP replica constantly processes `micro_batch_size` samples per micro-batch, 
        # only changing the number of micro-batches to maintain constant global batch size.
        micro_batch_size =  global_bs // num_micro_batches
        assert global_bs % (micro_batch_size * global_configs.data_parallel_size) == 0, \
            f"Global batch size {global_bs} must be divisible by micro-batch-size {micro_batch_size} " + \
            f"x data parallel size {global_configs.data_parallel_size}."
        num_micro_batches = global_bs // (micro_batch_size * global_configs.data_parallel_size)
        global_bs = micro_batch_size * num_micro_batches    # of this DP rank

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
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=(dtype == torch.float16),
            # MoE
            num_moe_experts=2,
            moe_router_topk=2,
        )
        # NOTE: Monkey patch all input configuratons into `megatron_config`, 
        # instead of explicitly pass-in into the arguments of `Trainer()`.
        setattr(megatron_config, "global_batch_size", global_bs)
        setattr(megatron_config, "num_micro_batches", num_micro_batches)
        setattr(megatron_config, "sequence_length", seq_len)
        setattr(megatron_config, "num_iters", num_iters)
        setattr(megatron_config, "dtype", dtype)
        # Monkey patch MoE specific configs due to lower Megatron version
        setattr(megatron_config, "moe_router_dtype", dtype)
        setattr(megatron_config, "moe_ffn_hidden_size", megatron_config.ffn_hidden_size)
        setattr(megatron_config, "moe_router_score_function", "softmax")
        setattr(megatron_config, "moe_router_enable_expert_bias", False)
        setattr(megatron_config, "moe_router_force_load_balancing", False)
        setattr(megatron_config, "moe_permute_fusion", False)
        setattr(megatron_config, "moe_apply_probs_on_input", False)
        setattr(megatron_config, "fp4", None)
        setattr(megatron_config, "fp8", None)

        # Model
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_seq_len": seq_len,
            "vocab_size": vocab_size,
            "device": device,
            "dtype": dtype,
            "megatron_config": megatron_config,
        }
        model = model_provider(**model_kwargs) if model_provider else None
        model = setup_model(model)
        
        train_data_iterator = get_train_data_iterator(
            vocab_size, global_bs, num_micro_batches, seq_len, num_iters=num_iters,
        )
        (optimizer, opt_param_scheduler) = get_optimizer_and_scheduler(
            model, global_bs, num_iters, fp16=(dtype == torch.float16), params_dtype=dtype, 
        )

        # Definition of forward step function
        def __forward_step_func(data_iterator, model):
            """ Vanilla forward step function. """
   
            def __loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
                """ Loss function definition. """
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                # If you have data parallel reduce loss across data parallel groups.
                # If pipeline parallel, loss computation is done only in last stage.
                return loss, {"lm loss": loss}

            megatron_config = get_attr_wrapped_model(model, attr="config")
            # Input batch
            (tokens, labels, loss_mask, 
             attention_mask, position_ids) = get_input_batch(
                 data_iterator=data_iterator,
                 micro_batch_size=(
                     megatron_config.global_batch_size // megatron_config.num_micro_batches
                 ),
                 seq_len=megatron_config.sequence_length,
                 pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            )
            # Forward
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor,  partial(__loss_func, loss_mask)

        # Trainer
        trainer = Trainer(
            megatron_transformer_config=megatron_config,
            forward_step_func=__forward_step_func,
            model=model,
            model_type=model_type,
            train_data_iterator=train_data_iterator,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            pipeline_strategy="1f1b",
            forward_only=False,
            use_pytorch_profiler=False,
        )
        trainer.train()


    # ===================> Deprecated tests
    @unittest.skip("Deprecated.")
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

    @unittest.skip("Deprecated.")
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
                "after_nodes": ['weight_grad', ],
                "interval": 1,
                "pass": 'backward',
                "parallelism": 'dp',
            },
            "all_reduce:tp:backward": {
                "after_markers": [],
                "after_nodes": ['weight_grad', ],
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
            parallelism_plan=parallelism_plan,
            only_forward=global_configs.only_forward,
            base_input_args=input_args,
            base_input_kwargs=input_kwargs,
            partition_pipeline_stages_func=CriusParallelPlanner.partition_pipeline_stages,
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

    @unittest.skip("Pass.")
    def test_meagtron_ground_truth(self):
        """ Run the parallelized Megatron model as the ground truth. """

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
        num_attention_heads = 32
        num_layers = 4
        data_dtype = torch.long
        dtype = torch.float16
        device = "cuda"
        vocab_size = 50000
        
        global_configs.training_backend = "megatron"
        global_configs.tensor_model_parallel_size = 2
        global_configs.pipeline_model_parallel_size = 1
        global_configs.data_parallel_size = 1
        global_configs.only_forward = False
        global_configs.enable_backward_graph_generation = True

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
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=True,
        )

        # Model
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
            },
            device_str=device,
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

        # Warmup
        for _ in range(100):
            with torch.no_grad():
                _ = model(*input_args, **input_kwargs)
            
            exit(0)
        
        torch.cuda.synchronize(device)
    
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        # Measure
        start_event.record()
        for _ in range(1000):
            with torch.no_grad():
                _ = model(*input_args, **input_kwargs)
        end_event.record()

        end_event.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / 1000     
        stage_latencies_ms = [avg_latency_ms]
        measured_e2e_latency =np.max(stage_latencies_ms) * (num_micro_batches - 1) + np.sum(stage_latencies_ms)
        
        print(f"========> Measured end-to-end latency: {measured_e2e_latency:.3f} ms")


if __name__ == '__main__':
    unittest.main()
