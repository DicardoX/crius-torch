#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MoE layer modules. """

from typing import Optional, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from crius.train_backend.megatron.moe.router import TopKRouter
from crius.train_backend.megatron.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from crius.train_backend.megatron.moe.experts import SequentialMLP

__all__ = [ "MoELayer", ]


class MoELayer(MegatronModule):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.

    TODO(chunyu): To support expert parallelism, we should implement token dispatcher.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        ep_size = 1     # TODO(chunyu): Currently we don't support expert parallelism.
        
        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size

        self.router = TopKRouter(self.config)

        # assert config.moe_token_dispather_type == "allgather", \
        #     "Only All-Gather dispatcher is supported currently."
        # self.token_dispatcher = MoEAllGatherTokenDispatcher(
        #     self.num_local_experts, self.local_expert_indices, self.config,
        # )
        self.token_dispatcher = None

        self.experts = SequentialMLP(self.num_local_experts, self.config)

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        
        probs, routing_map = self.router(hidden_states)

        # Reshapes hidden states
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Identifies tokens for local experts and permutes them for expert processing.
        self.hidden_shape_before_permute = hidden_states.shape
        self.local_map = routing_map.contiguous()   # TODO(chunyu): Currently no EP is supported.
        self.local_probs = probs.contiguous()
        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()
        (permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping) = permute(
            hidden_states, self.local_map, num_out_tokens=tokens_per_expert.sum(), 
            fused=self.config.moe_permute_fusion,
        )
        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        
        expert_output, mlp_bias = self.experts(permuted_local_hidden_states, tokens_per_expert, self.local_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        
        # Reverses token permutation to restore original ordering before reduction operations
        unpermuted_local_hidden = unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )

        # Restores the original tensor shape
        output = unpermuted_local_hidden.view(self.hidden_shape)

        return output, mlp_bias


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """

    assert not fused, "Fused permute is not supported yet."
    assert not drop_and_pad, "Drop and pad is not supported yet."

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    
    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    print(token_indices.shape, routing_map.shape, sorted_indices.shape)
    exit(0)

    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """

    assert not fused, "Fused unpermute is not supported yet."
    assert not drop_and_pad, "Drop and pad is not supported yet."

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )

    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )

    return output_tokens.to(dtype=input_dtype)
    