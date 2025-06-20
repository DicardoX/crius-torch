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

import torch
from torch import nn

from crius.loader import (ModelLoader, ShadowModelLoader)
from crius.core.profiler import CriusDistProfiler
from crius.model_parallel.utils import ParallelismPlan


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, device: str,):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, device=device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return self.fc2(x)


class ToyModel(nn.Module):
    """ Toy PyTorch model. """

    def __init__(
            self, hidden_size: int, num_layers: int, device: str,
    ) -> None:
        """ Initialize a toy model. """

        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList([ToyLayer(hidden_size, device) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, device=device)
        
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

    def test_torch_model_profiling(self):
        """ Profile a naive torch model. """
        
        global_bs = 4
        seq_len = 16
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
        
        # Profile
        profiler = CriusDistProfiler()
        profiler.profile(
            model=model,
            base_input_args=(data[0], ),
            parallelism_plan=ParallelismPlan(tensor_parallel_degree=2),
        )


if __name__ == '__main__':
    unittest.main()
