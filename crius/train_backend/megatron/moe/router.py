#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MoE router modules. """

from typing import Optional

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [ "TopKRouter", ]


class TopKRouter(MegatronModule):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts."""
    
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """

        super().__init__(config=config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
    
    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."

        # TODO(chunyu): We skip apply_z_loss here for compilation.
        assert self.score_function == "softmax", "Only softmax score function is supported currently."
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, self.topk, dim=1)
        probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        return probs, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # NOTE(chunyu): We skip input jitter to avoid randomness for compilation.
        logits = self.gating(input)
        assert not self.config.moe_router_force_load_balancing, "Not supported yet."
        probs, routing_map = self.routing(logits)
        return probs, routing_map


class RouterGatingLinearFunction(torch.autograd.Function):
    """
    Autograd function for router gating linear.
    """

    @staticmethod
    def forward(
        ctx, inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
    ):
        """
        Forward pass of the RouterGatingLinearFunction function.
        """
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        if bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(
                bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t()
            )

        output = output.view(*inp_shape[:-1], -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the RouterGatingLinearFunction function.
        """
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.view(-1, grad_shape[-1])

        grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
        grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)

        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        grad_input = grad_input.view(*inp_shape)
        return grad_input, grad_weight, grad_bias, None


def router_gating_linear(
    inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
):
    """
    Customized linear layer for router gating.
    This linear layer accepts bfloat16 input and weight, and can return output with router_dtype.
    It can reduce the memory usage by avoiding saving the intermediate high precision tensors.
    """
    return RouterGatingLinearFunction.apply(inp, weight, bias, router_dtype)
