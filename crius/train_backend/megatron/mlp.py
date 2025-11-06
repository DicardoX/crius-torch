#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MLP modules. """

from typing import Any, Dict, Optional, Union
import warnings

import torch
from torch import Tensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear
from crius.global_envs import global_configs

__all__ = [ "MLP", ]


class MLP(MegatronModule):
    """ MLP module with static submodules for compilation.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self, 
        config: TransformerConfig, 
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
    ) -> None:
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.input_size = input_size if input_size != None else self.config.hidden_size
        if ffn_hidden_size is None:
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size
        
        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = ColumnParallelLinear(
            self.input_size,
            ffn_hidden_size,
            tp_size=self.config.tensor_model_parallel_size,
            bias=self.config.add_bias_linear,
            # device=torch.cuda.current_device(),
            device=global_configs.current_device,
            dtype=self.config.params_dtype,
            skip_bias_add=True,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = RowParallelLinear(
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            tp_size=self.config.tensor_model_parallel_size,
            bias=self.config.add_bias_linear,
            # device=torch.cuda.current_device(),
            device=global_configs.current_device,
            dtype=self.config.params_dtype,
            input_is_parallel=True,
            skip_bias_add=True,
        )


    def forward(self, hidden_states, per_token_scale=None):
        """ Forward method for compilation. """

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)        

        # NOTE: Naive activation function for compilation.
        intermediate_parallel = self.activation_func(intermediate_parallel)

        if per_token_scale is not None:
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
            intermediate_parallel = intermediate_parallel.to(original_dtype)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None    

        return output, output_bias
