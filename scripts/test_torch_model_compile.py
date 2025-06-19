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

from crius.compile import (compile, CompileConfigs)


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
        self.layers = [ToyLayer(hidden_size, device) for _ in range(num_layers)]
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


class TorchModelCompileTest(unittest.TestCase):
    """ Basic torch model compilation test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    def test_torch_model_compilation(self):
        """ Compile a naive torch model into IR and parse its computation operators. """
        
        global_bs = 4
        seq_len = 4
        num_micro_batches = 2
        hidden_size = 4
        num_layers = 4
        dtype = torch.float16
        device = "cuda:0"

        # Model
        model = ToyModel(hidden_size, num_layers, device)
        # Inputs
        data = [
            torch.randn(global_bs, seq_len, hidden_size, dtype=dtype, device=device)
                for _ in range(num_micro_batches)
        ]
        
        # Compile 
        compile_configs = CompileConfigs(
            base_input_args=(data[0], ),
            parallelism_plan="tensor",
            parallelism_degree=2,
        )
        compiled_model = compile(model, compile_configs)



if __name__ == '__main__':
    unittest.main()
