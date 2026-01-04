#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MoE expert modules. """

from copy import deepcopy

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from crius.train_backend.megatron.mlp import MLP

__all__ = [ "SequentialMLP" ]


class SequentialMLP(MegatronModule):
    """A compilable implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
    ):
        
        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)
        
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
            )
            self.local_experts.append(expert)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""
        
        assert not self.config.moe_apply_probs_on_input, "Not supported yet."
        if self.num_local_experts == 1:
            assert not (self.config.fp8 or self.config.fp4), "Not supported yet."
            output, output_bias = self.local_experts[0](
                permuted_local_hidden_states, permuted_probs
            )
            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)
            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                assert not (self.config.fp8 or self.config.fp4), "Not supported yet."
                output, output_bias = expert(tokens, probs)
                output_local_list.append(output)
            
            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local
