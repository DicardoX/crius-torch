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
from crius.model_parallel.utils import ParallelismPlan
from crius.core.tuner import CriusParallelTuner
from crius.global_envs import global_configs, logger


class ParallelTuneTest(unittest.TestCase):
    """ Parallelism optimization test cases. 
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    def test_megatron_parallel_optimization(self):
        """ Optimize parallelism plans. """

        # from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.module import MegatronModule
        from megatron.core import mpu
        # from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from crius.train_backend.megatron.utils import initialize_distributed, initialize_virtual_distributed
        from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
        from crius.train_backend.megatron.transformer_layer import TransformerLayer
        from crius.train_backend.megatron.gpt_model import GPTModel

        def model_provider(
            hidden_size: int, num_layers: int, max_seq_len: int, vocab_size: int, device: str, dtype: str, 
            megatron_config: Any = None,
        ) -> nn.Module:
            """ Model provider function. """
            return GPTModel(megatron_config, vocab_size, max_seq_len, device=device)

        global_bs = 8
        seq_len = 1024
        num_micro_batches = 2
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 16
        data_dtype = torch.long
        dtype = torch.float16
        # device = "cuda"
        device = "cpu"
        vocab_size = 50000
        
        global_configs.num_nodes = 1
        global_configs.num_devices_per_node = 2
        global_configs.training_backend = "megatron"
        global_configs.only_forward = False
        global_configs.enable_backward_graph_generation = True
        if device == "cpu":
            logger.warning(
                "Shape propagation currently hangs in CPU mode, or prabably due to extremely slow progagation. " +
                "Currently we recommend using shadow model loader to reduce device memory consumption."
            )
            device = "mlu"
            global_configs.current_device = torch.cuda.current_device()
            global_configs.use_shadow_model_loader = True

        # Prototype of parallelism plan
        pipeline_stage_partition_method = "auto"
        # pipeline_stage_partition_method = "uniform"
        pipeline_use_boundary_markers = True
        pipeline_boundary_start_marker = ['input_layernorm', ]
        pipeline_boundary_end_marker = ['final_layernorm', ]
        proto_parallelism_plan = ParallelismPlan(
            pipeline_type="1f1b",
            num_layers=num_layers,
            num_attn_heads=num_attention_heads,
            num_microbatches=num_micro_batches,
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
            # tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            # pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_nodes=global_configs.num_nodes,
            num_devices_per_node=global_configs.num_devices_per_node,
            init_cpu_process_group=True,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron
        megatron_config = TransformerConfig(
            # tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            # pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
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

        # Tune
        tuner = CriusParallelTuner(
            model=model,
            num_nodes=global_configs.num_nodes,
            num_devices_per_node=global_configs.num_devices_per_node,
            proto_parallelism_plan=proto_parallelism_plan,
            grid_plan_optimization_mode="all",
            profiled_data_path='./crius/offline/prof_data.pkl',
        )
        optimal_plan, least_e2e_latency = tuner.tune(input_args, input_kwargs)

        tuner = CriusParallelTuner(
            model=model,
            num_nodes=global_configs.num_nodes,
            num_devices_per_node=global_configs.num_devices_per_node,
            proto_parallelism_plan=proto_parallelism_plan,
            grid_plan_optimization_mode="all",
            profiled_data_path='./crius/offline/prof_data.pkl',
        )
        optimal_plan, least_e2e_latency = tuner.tune(input_args, input_kwargs)

        print(f"\n\n\n================================")
        print(f"=     Optimization results     =")
        print(f"================================")
        print(f"\n====> Optimal parallelism plan:")
        print(f"\n - Tensor parallel degree: {optimal_plan.tensor_parallel_degree}")
        print(f"\n - Data parallel degree: {optimal_plan.data_parallel_degree}")
        print(f"\n - Pipeline parallel degree (i.e., number of pipeline stages): {len(optimal_plan.pipeline_num_layers_per_stage)}")
        print(f"\n - Pipeline stage partition ('x' denotes a single transformer layer):")
        stage_names = [f"Stage {i+1} ({optimal_plan.pipeline_num_layers_per_stage[i]} layers)" for i in range(len(optimal_plan.pipeline_num_layers_per_stage))]
        name_line = " | ".join(stage_names)
        str_ = ""
        for i, n in enumerate(optimal_plan.pipeline_num_layers_per_stage):
            str_ = str_ + f"[{' '.join(['x' for _ in range(n)])}] | " if i < len(optimal_plan.pipeline_num_layers_per_stage) - 1 else str_ + f"[{' '.join(['x' for _ in range(n)])}]"
        print("   ", name_line)
        print("   ", str_)
        print(f"\n - Metadata: {optimal_plan}")

        print(f"\n====> Latency of the optimal plan (ms):")
        print(f"\n - {least_e2e_latency}")


if __name__ == '__main__':
    unittest.main()
