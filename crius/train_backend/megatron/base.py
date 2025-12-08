#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
import torch
from torch import Tensor
import torch.nn.functional as F


__all__ = [
    "ModelType",
    "AttnMaskType",
    "SimpleTransformerConfig",
    "SimpleLanguageModule",
    "get_bias_dropout_add",
    "make_viewless_tensor",
    "MLPSubmodules",
]


class ModelType(Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class AttnMaskType(Enum):
    padding = 1
    causal = 2
    no_mask = 3


@dataclass
class SimpleTransformerConfig:
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_query_groups: int = 12
    ffn_hidden_size: int = 3072
    kv_channels: int = 64
    
    activation_func: Callable = F.gelu
    gated_linear_unit: bool = False
    layernorm_epsilon: float = 1e-5
    
    add_bias_linear: bool = True
    add_qkv_bias: bool = False
    
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False
    
    bias_dropout_fusion: bool = False
    masked_softmax_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    
    num_moe_experts: Optional[int] = None
    moe_router_topk: int = 2
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 0.01
    moe_token_dispatcher_type: str = "allgather"
    moe_grouped_gemm: bool = False
    moe_router_pre_softmax: bool = True
    
    def __post_init__(self):
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        
        if self.bf16:
            self.params_dtype = torch.bfloat16
        elif self.fp16:
            self.params_dtype = torch.float16


class SimpleLanguageModule(torch.nn.Module):
    
    def __init__(self, config: SimpleTransformerConfig):
        super().__init__()
        self.config = config
    
    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        raise NotImplementedError("Subclass must implement set_input_tensor")
    
    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        labels = labels.transpose(0, 1).contiguous()
        logits = logits.transpose(1, 2).contiguous()
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return loss


def make_viewless_tensor(
    inp: Tensor,
    requires_grad: bool,
    keep_graph: bool,
) -> Tensor:
    return inp


def get_bias_dropout_add(training: bool, fused: bool):
    def bias_dropout_add(x_with_bias, residual, dropout_prob):
        if isinstance(x_with_bias, tuple):
            x, bias = x_with_bias
        else:
            x = x_with_bias
            bias = None
        
        if bias is not None:
            x = x + bias
        
        if training and dropout_prob > 0.0:
            x = F.dropout(x, p=dropout_prob, training=training)
        
        return x + residual
    
    return bias_dropout_add


@dataclass
class MLPSubmodules:
    linear_fc1: Any = None
    linear_fc2: Any = None
