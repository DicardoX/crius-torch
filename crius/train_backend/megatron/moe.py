#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear


__all__ = [
    "TopKRouter",
    "MoEExpert",
    "SimpleMoELayer",
]


class TopKRouter(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        self.weight = nn.Parameter(
            torch.empty(hidden_size, num_experts, device=device, dtype=dtype)
        )
        
        nn.init.kaiming_uniform_(self.weight, a=1.0)
    
    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        original_shape = hidden_states.shape
        
        if hidden_states.dim() == 3:
            seq_len, batch_size, hidden_size = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_size)
        else:
            hidden_states_flat = hidden_states
        
        router_logits = torch.matmul(hidden_states_flat, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        expert_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        expert_weights = expert_weights.to(hidden_states_flat.dtype)
        
        aux_loss: Optional[Tensor] = None
        if self.training and self.aux_loss_coeff > 0:
            aux_loss = self._compute_aux_loss(router_probs, expert_indices)
        
        return expert_weights, expert_indices, router_logits, aux_loss
    
    def _compute_aux_loss(
        self,
        router_probs: Tensor,
        expert_indices: Tensor,
    ) -> Tensor:
        num_tokens = router_probs.size(0)
        
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
        if self.top_k > 1:
            expert_mask = expert_mask.sum(dim=1)
        expert_mask = expert_mask.float()
        
        tokens_per_expert = expert_mask.sum(dim=0)
        f = tokens_per_expert / (num_tokens * self.top_k)
        
        P = router_probs.mean(dim=0)
        
        aux_loss = self.num_experts * torch.sum(f * P)
        
        return aux_loss * self.aux_loss_coeff


class MoEExpert(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation_func: Callable = F.silu,
        gated_linear_unit: bool = True,
        add_bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation_func = activation_func
        self.gated_linear_unit = gated_linear_unit
        
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        fc1_output_size = ffn_hidden_size * 2 if gated_linear_unit else ffn_hidden_size
        
        self.linear_fc1 = nn.Linear(
            hidden_size, fc1_output_size, bias=add_bias, device=device, dtype=dtype
        )
        
        self.linear_fc2 = nn.Linear(
            ffn_hidden_size, hidden_size, bias=add_bias, device=device, dtype=dtype
        )
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        intermediate = self.linear_fc1(hidden_states)
        
        if self.gated_linear_unit:
            gate, value = torch.chunk(intermediate, 2, dim=-1)
            intermediate = self.activation_func(gate) * value
        else:
            intermediate = self.activation_func(intermediate)
        
        output = self.linear_fc2(intermediate)
        
        return output


class SimpleMoELayer(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        activation_func: Callable = F.silu,
        gated_linear_unit: bool = True,
        add_bias: bool = False,
        aux_loss_coeff: float = 0.01,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coeff=aux_loss_coeff,
            dtype=dtype,
            device=device,
        )
        
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                activation_func=activation_func,
                gated_linear_unit=gated_linear_unit,
                add_bias=add_bias,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        original_shape = hidden_states.shape
        seq_len, batch_size, hidden_size = original_shape
        
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states_flat.size(0)
        
        expert_weights, expert_indices, _, _ = self.router(hidden_states_flat)
        
        output = torch.zeros_like(hidden_states_flat)
        
        expert_one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            
            expert_output = expert(hidden_states_flat)
            
            weight_for_expert = (expert_weights * expert_one_hot[:, :, expert_idx]).sum(dim=1)
            
            output = output + expert_output * weight_for_expert.unsqueeze(-1)
        
        output = output.view(*original_shape)
        
        return output, None
    
    def compute_aux_loss(self, hidden_states: Tensor) -> Optional[Tensor]:
        if not self.training or self.aux_loss_coeff <= 0:
            return None
        
        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        else:
            hidden_states_flat = hidden_states
        
        router_logits = torch.matmul(hidden_states_flat, self.router.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        _, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        return self.router._compute_aux_loss(router_probs, expert_indices)


def create_moe_layer_from_config(config: Any, layer_number: Optional[int] = 1) -> SimpleMoELayer:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    return SimpleMoELayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.ffn_hidden_size,
        num_experts=config.num_moe_experts,
        top_k=getattr(config, 'moe_router_topk', 2),
        activation_func=getattr(config, 'activation_func', F.silu),
        gated_linear_unit=getattr(config, 'gated_linear_unit', True),
        add_bias=getattr(config, 'add_bias_linear', False),
        aux_loss_coeff=getattr(config, 'moe_aux_loss_coeff', 0.01),
        dtype=config.params_dtype,
        device=device,
    )
