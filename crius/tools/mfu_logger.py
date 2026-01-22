#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Toolkit for logging MFU metric. """

from typing import Union, List, Optional
from dataclasses import dataclass

__all__ = [
    "MFULoggerModelConfig",
    "MFULogger",
]


@dataclass
class MFULoggerModelConfig:
    """ Model configuration class for MFU Logger. 
    
    This class behaves as a subset of Megatron's TransformerConfig.
    """

    num_layers: int = None
    """ Number of layers. """

    num_attention_heads: int = None
    """ Number of attention heads. """

    hidden_size:int = None
    """Transformer hidden size."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """ Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided. """

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    ###############
    # MoE-related #
    ###############
    num_experts: Optional[int] = None
    """ Number of experts. """

    moe_layer_freq: Union[int, List[int]] = 1
    """ Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0] """

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_shared_expert_intermediate_size: Optional[int] = None
    """ Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert.
    By default, the shared experts execute before the router. However, when
    moe_shared_expert_overlap or overlap_moe_expert_parallel_comm is set,
    the shared experts execute after the router, before the routed experts.
    This makes the gradients from the router and the shared experts added in
    different orders to the hidden_states, causing minor numerical differences
    in the hidden_states gradient. """

    ##########
    # Others #
    ##########
    swiglu: bool = False
    """ Use swiglu activation function. """

    mtp_num_layers: int = None
    """ Number of multi-token prediction layers. """

    attention_output_gate: bool = False
    """Whether to apply output gate to the attention layers."""

    def __post_init__(self):
        """ Post init. """
        
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads


class MFULogger:
    """ MFU logger class. """

    def __init__(
        self, model_config: MFULoggerModelConfig,
    ) -> None:
        """ Initialize a MFU logger. """
        self.config: MFULoggerModelConfig = model_config

    def num_transformer_floating_point_operations(self, batch_size: int, sequence_length: int) -> int:
        """ 
        Calculate the number of floating point operations for a standard Transformer model. 
        Refer: `Megatron/megatron/training/training/num_floating_point_operations`.
        """

        if self.config.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = self.config.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(self.config.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % self.config.moe_layer_freq == 0) else 0 for i in range(self.config.num_layers)
                ]
            elif isinstance(self.config.moe_layer_freq, list):
                moe_layer_pattern = self.config.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == self.config.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {self.config.num_layers}, "
                f"current moe layer pattern: {self.config.moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = self.config.num_layers - num_moe_layers
            num_experts_routed_to = self.config.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]
        
        if mtp_num_layers is None:
            mtp_num_layers = 0
            num_layers = self.config.num_layers
        else:
            assert NotImplementedError
        
        moe_ffn_hidden_size = (
            self.config.moe_ffn_hidden_size
            if self.config.moe_ffn_hidden_size is not None
            else self.config.ffn_hidden_size
        )
        shared_expert_ffn_hidden_size = (
            0
            if self.config.moe_shared_expert_intermediate_size is None
            else self.config.moe_shared_expert_intermediate_size
        )

        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        forward_backward_expansion_factor = 3
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        fma_expansion_factor = 2
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if self.config.swiglu else 2
        
        ## MHA or GQA
        query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        key_projection_size = self.config.kv_channels * self.config.num_query_groups
        value_projection_size = self.config.kv_channels * self.config.num_query_groups
        gate_projection_size = query_projection_size if self.config.attention_output_gate else 0
        standard_self_attn_term = (
            forward_backward_expansion_factor
            * fma_expansion_factor
            * (
                ## qkv proj
                self.config.hidden_size
                * (
                    query_projection_size
                    + key_projection_size
                    + value_projection_size
                    + gate_projection_size
                )
                ## core attention
                + query_projection_size
                * sequence_length
                / 2  # causal mask (only half of the mask is non-zero)
                * 2  # QK^T and (QK^T)V
                ## out proj
                + query_projection_size
                * self.config.hidden_size
            )
        )

        raise NotImplementedError
        