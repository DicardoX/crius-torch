#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's attention modules. """

from typing import Any, Dict, Optional, Union, Tuple

import torch
from torch import Tensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.enums import AttnMaskType

from crius.train_backend.megatron.layers import ColumnParallelLinear

__all__ = [ "SelfAttention", ]


class SelfAttention(MegatronModule):
    """ Self-attention layer with static submodules for compilation.

    Args:
        config: Transformer configuration in Megatron.
    """

    def __init__(
        self, 
        config: TransformerConfig, 
        layer_number: Optional[int] = None,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = 'self',
    ) -> None:
        super().__init__(config=config)

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values
        world_size = self.config.tensor_model_parallel_size
        assert self.config.num_attention_heads % world_size == 0
        self.hidden_size_per_attention_head = self.query_projection_size // self.config.num_attention_heads
        assert self.config.num_attention_heads % world_size == 0
        self.num_attention_heads_per_partition = self.config.num_attention_heads // world_size
        assert self.config.num_query_groups % world_size == 0
        self.num_query_groups_per_partition = self.config.num_query_groups // world_size
        
        self.linear_qkv = ColumnParallelLinear(
            input_size=self.config.hidden_size, 
            output_size=(self.query_projection_size + 2 * self.kv_projection_size),
            tp_size=self.config.tensor_model_parallel_size,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
            skip_bias_add=False,
        )
        self.q_layernorm = None     # NOTE: IdentityOp() cannot be compiled 
        self.k_layernorm = None

        self.core_attention = DotProductAttention(
            self.config, layer_number, attn_mask_type, attention_type,
        )
    
    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, split_qkv=True):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`. If `split_qkv=False`, then
        the unsplit mixed_qkv tensor is returned.
        """

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # Return unsplit mixed_qkv and split_arg_list
        if not split_qkv:
            return mixed_qkv, split_arg_list
        
        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)
        
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """ Forward method for compilation.
        
        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.
        """
        split_qkv = True

        # Query, key, and value
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True,    # TODO(chunyu): Current split qkv for self-attn
        )

        if split_qkv:
            query, key, value = qkv_output
        else:
            raise NotImplementedError("Currently only split_qkv=True is supported.")
        
        # # Core attention computation
        # core_attn_out = self.core_attention(
        #     query, key, value, attention_mask, attn_mask_type=self.attn_mask_type,
        # )

        # return core_attn_out
        return query
