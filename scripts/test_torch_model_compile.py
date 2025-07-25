#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set, Union
from typing import Optional, Union, List, Callable
import sympy as sp
import unittest
import logging
from functools import partial
from collections import OrderedDict
import time

import torch
from torch import nn

from crius.loader import (ModelLoader, ShadowModelLoader)
from crius.core.profiler import CriusDistProfiler
from crius.model.simple_gpt import SimpleGPTModel, SimpleGPTConfig
from crius.model.tf_gpt_model import GPTModel, GPTConfig
from crius.compile.graph.transform import (transform_to_ir_graph, print_model_graph, 
        analyze_graph_performance, print_operator_summary,count_core_function_ops,
        count_core_module_ops, call_function_core_op_category, call_module_core_op_category2, 
        count_core_method_ops)
from crius.model_parallel.utils import ParallelismPlan
from crius.compile.compiler import CompileConfigs, compile


class TorchModelCompileTest(unittest.TestCase):
    @unittest.skip("Pass.")
    def test_simple_gpt_model_compiling(self):

        global_bs = 4
        seq_len = 16
        num_micro_batches = 2
        hidden_size = 128
        num_layers = 4
        dtype = torch.long  
        device = torch.device("cpu") 

        def model_provider(**kwargs) -> nn.Module:
            config = SimpleGPTConfig(
                    vocab_size=50257,
                    hidden_size=768,
                    num_layers=12,
                    num_heads=12,
                    dropout=0.1
            )
            return SimpleGPTModel(config)

        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider,
            model_kwargs={
                "device": device,
            },
            fake_device_str=device,
        )
        model = model_loader.load()
        model.to(device)

        input_ids = torch.randint(0, 50257, (global_bs // num_micro_batches, seq_len), dtype=dtype).to(device)
        graph = transform_to_ir_graph(model, input_ids)
        print(graph)
        print("-------------------")
        print_model_graph(graph)    
        operator_counts = analyze_graph_performance(graph)
        print_operator_summary(operator_counts)      

    def test_normal_gpt_model_compiling(self):

        global_bs = 4
        seq_len = 16
        num_micro_batches = 2
        dtype = torch.long  
        device = torch.device("cpu") 

        def model_provider2(**kwargs) -> nn.Module:
            config = GPTConfig(
                    vocab_size=50257,
                    n_positions= 1024,
                    n_layer =12,
                    n_head=12,
            )
            return GPTModel(config)

        _model_loader_class = ShadowModelLoader
        model_loader = _model_loader_class(
            model_provider=model_provider2,
            model_kwargs={
                "device": device,
            },
            fake_device_str=device,
        )
        model = model_loader.load()
        model.to(device)

        input_ids = torch.randint(0, 50257, (global_bs // num_micro_batches, seq_len), dtype=dtype).to(device)
        graph = transform_to_ir_graph(model, input_ids)
        print(graph)
        print("-------------------")
        print_model_graph(graph)
        operator_counts = analyze_graph_performance(graph)
        print_operator_summary(operator_counts)  

        print("\n\n===========================")
        op_counter = count_core_module_ops(graph)   
        print("\n\n=======>Call Module Core op <=======")
        for op_name, count in sorted(op_counter.items()):
            print(f"{op_name:<20} <---> {call_module_core_op_category2[op_name]:<20} : {count}")

        op_counter = count_core_function_ops(graph)   
        print("\n\n=======>Call Function Core op <=======")
        for op_name, count in sorted(op_counter.items()):
            print(f"{op_name:<20} <---> {call_function_core_op_category[op_name]:<50} : {count}")

        op_counter = count_core_method_ops(graph) 
        print("\n\n=======>Call Method Core op <=======")
        for op_name, count in sorted(op_counter.items()):
            print(f"{op_name:<20} : {count}")


if __name__ == '__main__':
    unittest.main()
