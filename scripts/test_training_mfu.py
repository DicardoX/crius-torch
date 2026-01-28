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


class TrainMFUTest(unittest.TestCase):
    """ Training MFU test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    @unittest.skip("Pass.")
    def test_one_torch_operator(self):
        """ Profile one torch operator in a hackable manner. """

        num_warmup_steps = 5
        num_profile_steps = 100
        use_torch_empty_to_init_weight = True
        batch_size = 4
        seq_len = 1024
        hidden_size = 4096
        ffn_hidden_size = 4 * 4096

        # GEMM
        left_op = torch.empty((seq_len, batch_size, hidden_size), device="cuda", dtype=torch.float16) \
            if use_torch_empty_to_init_weight else torch.randn((seq_len, batch_size, hidden_size), device="cuda", dtype=torch.float16)
        right_op = torch.empty((ffn_hidden_size, hidden_size), device="cuda", dtype=torch.float16) \
            if use_torch_empty_to_init_weight else torch.randn((ffn_hidden_size, hidden_size), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

        for _ in range(num_warmup_steps):
            _ = torch.matmul(left_op, right_op.t())
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)  
        # Measure
        start_event.record()
        for _ in range(num_profile_steps):
            _ = torch.matmul(left_op, right_op.t())
        
        end_event.record()
        end_event.synchronize()

        op_latency = (start_event.elapsed_time(end_event) / num_profile_steps) / 1e3    # ms
        token_thr = (batch_size * seq_len) / op_latency
        op_flops = 2 *  batch_size * seq_len * hidden_size * ffn_hidden_size
        flops_per_token = op_flops / (batch_size * seq_len)
        train_tflops = (flops_per_token * token_thr) / 1e12
        print(f"Operator latency (s): {(op_latency):.3f} | Model FLOPs per token: {flops_per_token} | " + 
              f"Training TFLOPS: {train_tflops}")

    # @unittest.skip("Pass.")
    def test_meagtron_train_mfu_gpt(self):
        """ Run the parallelized Megatron model, using Megatron's parallelization codes. """

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
            megatron_config: Any, max_seq_len: int, vocab_size: int, device: str,
        ) -> nn.Module:
            """ Model provider function. """
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            return GPTModel(megatron_config, vocab_size, max_seq_len, pre_process, post_process, device=device)

        def megatron_native_model_provider():
            """ Megatron-native model provider function. """
            raise NotImplementedError

        model_type = ModelType.GPT
        global_bs = 8
        seq_len = 1024
        num_micro_batches = 4
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

        global_configs.num_nodes = 1
        global_configs.num_devices_per_node = 2
        global_configs.peak_fp16_tflops = 149.7     # NVIDIA A40

        # Transformer engine
        # NOTE: If error: AttributeError: module 'transformer_engine' has no attribute 'pytorch'
        # Use the following commands to reinstall TE:
        # ```
        # MAX_JOBS=8 pip3 install --no-build-isolation transformer-engine[pytorch]
        # pip3 install pyyaml
        # ```
        # NOTE: If error: AssertionError: CUDA device, driver and/or toolkit version does not 
        # support comm+GEMM overlap with CUDA Multicast. Launch app with UB_SKIPMC=1 to try CUDA IPC instead.
        # This is because: CUDA multicast was added in CUDA 12.1.0 and the minimum hardware requirements are 
        # SM 9.0 (Hopper), NVSwitch 3, NVLink 4 (see requirements for SHARP).
        # Use the following commands to use CUDA IPC.
        # ```
        # export UB_SKIPMC=1
        # ```
        os.environ["USE_TRANSFORMER_ENGINE"] = "True"
        tp_comm_overlap = True
        global_configs.sequence_parallel = True

        # Megatron-native pipeline optimizations
        virtual_pipeline_model_parallel_size = None
        overlap_p2p_comm = False
        batch_p2p_comm = True
        batch_p2p_sync = True
        
        # Get number of microbatches
        # In Megatron, each DP replica constantly processes `micro_batch_size` samples per micro-batch, 
        # only changing the number of micro-batches to maintain constant global batch size.
        micro_batch_size =  global_bs // num_micro_batches
        assert global_bs % (micro_batch_size * global_configs.data_parallel_size) == 0, \
            f"Global batch size {global_bs} must be divisible by micro-batch-size {micro_batch_size} " + \
            f"x data parallel size {global_configs.data_parallel_size}."
        num_micro_batches = global_bs // (micro_batch_size * global_configs.data_parallel_size)
        global_bs = micro_batch_size * num_micro_batches    # of this DP rank

        global_configs.micro_batch_size = micro_batch_size
        global_configs.sequence_length = seq_len
        global_configs.hidden_size = hidden_size

        # Init distributed
        initialize_distributed(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            tp_comm_overlap=tp_comm_overlap,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron   
        megatron_config = TransformerConfig(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            sequence_parallel=global_configs.sequence_parallel,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=(dtype == torch.float16),
            tp_comm_overlap=tp_comm_overlap,
            tp_comm_overlap_ag=True,
            tp_comm_overlap_rs=True,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            batch_p2p_sync=batch_p2p_sync,
        )

        # NOTE: We monkey patch all input configuratons into `megatron_config`, 
        # instead of explicitly pass-in into the arguments of `Trainer()`.
        setattr(megatron_config, "global_batch_size", global_bs)
        setattr(megatron_config, "vocab_size", vocab_size)
        setattr(megatron_config, "num_micro_batches", num_micro_batches)
        setattr(megatron_config, "sequence_length", seq_len)
        setattr(megatron_config, "num_iters", num_iters)
        setattr(megatron_config, "dtype", dtype)

        # Model
        model_kwargs={
            "megatron_config": megatron_config,
            "max_seq_len": seq_len,
            "vocab_size": vocab_size,
            "device": device,
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
            pytorch_profiler_dp_ranks=[0, ], 
            pytorch_profiler_tp_ranks=[0, ],
        )
        trainer.train(metric_types=["e2e_iter_time", "mfu", ])


if __name__ == '__main__':
    unittest.main()
