#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Optional, Union
from dataclasses import dataclass, field
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

from crius.train_backend.megatron.layers import ColumnParallelLinear, LanguageModelEmbedding
from crius.train_backend.megatron.llama_model import RMSNorm
from crius.train_backend.megatron.moe import SimpleMoELayer, create_moe_layer_from_config
from crius.train_backend.megatron.base import (
    ModelType,
    SimpleTransformerConfig,
    SimpleLanguageModule,
    get_bias_dropout_add,
    make_viewless_tensor,
)
from crius.train_backend.megatron.pipeline_parallel import schedules, p2p_communication
from crius.global_envs import logger


__all__ = ["QwenMoEModel", "QwenMoETransformerLayer", "QwenMoETransformerBlock"]


class QwenMoETransformerLayer(torch.nn.Module):

    def __init__(
        self,
        config,
        layer_number: Optional[int] = None,
        hidden_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        from crius.train_backend.megatron.attention import SelfAttention

        self.config = config
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        if not hasattr(config, 'num_moe_experts') or config.num_moe_experts is None:
            raise ValueError("QwenMoETransformerLayer requires config.num_moe_experts to be set")

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.params_dtype,
        )

        self.self_attention = SelfAttention(config, layer_number)
        self.self_attn_bda = get_bias_dropout_add

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.params_dtype,
        )

        self.moe = create_moe_layer_from_config(config, layer_number)

        self.mlp_bda = get_bias_dropout_add
        self.bias_dropout_add_exec_handler = torch.enable_grad
        self._make_viewless_tensor = make_viewless_tensor

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, getattr(self.config, 'bias_dropout_fusion', False))(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        moe_output_with_bias = self.moe(hidden_states)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, getattr(self.config, 'bias_dropout_fusion', False))(
                moe_output_with_bias, residual, self.hidden_dropout
            )

        output = self._make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, None


class QwenMoETransformerBlock(torch.nn.Module):

    def __init__(
        self,
        config,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__()
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        
        self.input_tensor = None

        self.num_layers_per_pipeline_rank = config.num_layers
        self._build_layers()

        if self.post_process:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                device=device,
                dtype=config.params_dtype,
            )

    def _build_layers(self):
        self.layers = torch.nn.ModuleList(
            [
                QwenMoETransformerLayer(
                    config=self.config,
                    layer_number=i + 1,
                )
                for i in range(self.num_layers_per_pipeline_rank)
            ]
        )

    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
    ):
        if hidden_states is None:
            hidden_states = self.input_tensor

        for layer in self.layers:
            hidden_states, context = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
            )

        if self.post_process:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class QwenMoEModel(torch.nn.Module):

    def __init__(
        self,
        config,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not hasattr(config, 'num_moe_experts') or config.num_moe_experts is None:
            raise ValueError(
                "QwenMoEModel requires config.num_moe_experts to be set. "
                "Please specify the number of experts in your config."
            )
        
        if hasattr(config, 'gated_linear_unit') and not config.gated_linear_unit:
            warnings.warn(
                "Qwen models use SwiGLU activation (gated_linear_unit=True). "
                "Setting gated_linear_unit=False will not match Qwen architecture.",
                UserWarning
            )

        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.max_position_embeddings = max_sequence_length
        self.rope_theta = rope_theta

        self.position_embedding_type = 'rope'
        self.model_type = ModelType.encoder_or_decoder

        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type='rope',
            )

        self.decoder = QwenMoETransformerBlock(
            config=self.config,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.post_process:
            self.output_layer = ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config.tensor_model_parallel_size,
                bias=False,
                device=self.device,
                dtype=self.config.params_dtype,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )

        logger.info(
            f"Initialized QwenMoEModel with vocab_size={vocab_size}, "
            f"max_seq_length={max_sequence_length}, rope_theta={rope_theta}, "
            f"num_experts={config.num_moe_experts}, "
            f"top_k={getattr(config, 'moe_router_topk', 2)}"
        )

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        if input_tensor is None:
            return
        
        input_tensor_list = [input_tensor] if not isinstance(input_tensor, list) else input_tensor

        assert len(input_tensor_list) == 1, 'input_tensor should only be length 1 for Qwen MoE'
        self.decoder.set_input_tensor(input_tensor_list[0])

    def _preprocess(
        self,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor],
        decoder_input: Optional[Tensor] = None,
    ) -> Optional[Tensor]:

        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            decoder_input = None
        
        if decoder_input is not None and decoder_input.dtype != self.config.params_dtype:
            decoder_input = decoder_input.to(self.config.params_dtype)

        return decoder_input

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:

        decoder_input = self._preprocess(input_ids, position_ids, decoder_input)

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
        )

        return self._postprocess(hidden_states, labels, runtime_gather_output)

    def _postprocess(
        self,
        hidden_states: Tensor,
        labels: Optional[Tensor],
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if not self.post_process:
            return hidden_states
        
        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss
    
    def shared_embedding_or_output_weight(self) -> Optional[Tensor]:
        if self.pre_process:
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        labels = labels.transpose(0, 1).contiguous()
        logits = logits.transpose(1, 2).contiguous()    
        torch_loss_func = torch.nn.CrossEntropyLoss()
        loss = torch_loss_func(logits, labels)

        return loss
    
    def get_forward_backward_func(self):
        return schedules.get_forward_backward_func()
    
    def pipeline_forward_backward(
        self,
        data_iterator,
        optimizer,
        num_microbatches: int,
        forward_only: bool = False,
    ):
        forward_backward_func = self.get_forward_backward_func()
        
        def forward_step(model, data_iterator, input_tensor, forward_only):
            return schedules.forward_step(
                model=model,
                data_iterator=data_iterator,
                input_tensor=input_tensor,
                forward_only=forward_only,
            )
        
        losses = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self,
            optimizer=optimizer,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
        )
        
        return losses
    
    def compute_moe_aux_loss(self, hidden_states_list: list) -> Optional[Tensor]:
        if not self.training:
            return None
        
        total_aux_loss: Optional[Tensor] = None
        for layer_idx, layer in enumerate(self.decoder.layers):
            if hasattr(layer, 'moe') and layer_idx < len(hidden_states_list):
                moe_layer = layer.moe
                if isinstance(moe_layer, SimpleMoELayer):
                    aux_loss = moe_layer.compute_aux_loss(hidden_states_list[layer_idx])
                    if aux_loss is not None:
                        if total_aux_loss is None:
                            total_aux_loss = aux_loss
                        else:
                            total_aux_loss = total_aux_loss + aux_loss
        return total_aux_loss


def qwen2_5_moe_a14b_config(**kwargs) -> SimpleTransformerConfig:
    config_dict = dict(
        num_layers=28,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=5632,
        kv_channels=128,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        layernorm_epsilon=1e-6,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.01,
        expert_model_parallel_size=1,
    )
    config_dict.update(kwargs)
    return SimpleTransformerConfig(**config_dict)


def qwen2_5_moe_57b_a14b_config(**kwargs) -> SimpleTransformerConfig:
    config_dict = dict(
        num_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_query_groups=4,
        ffn_hidden_size=18944,
        kv_channels=128,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        layernorm_epsilon=1e-6,
        num_moe_experts=64,
        moe_router_topk=8,
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.001,
        expert_model_parallel_size=1,
    )
    config_dict.update(kwargs)
    return SimpleTransformerConfig(**config_dict)


def qwen2_5_moe_custom_config(
    num_layers: int = 28,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_query_groups: int = 16,
    ffn_hidden_size: int = 5632,
    num_experts: int = 8,
    top_k: int = 2,
    **kwargs
) -> SimpleTransformerConfig:
    config_dict = dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        kv_channels=hidden_size // num_attention_heads,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        layernorm_epsilon=1e-6,
        num_moe_experts=num_experts,
        moe_router_topk=top_k,
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.01,
        expert_model_parallel_size=1,
    )
    config_dict.update(kwargs)
    return SimpleTransformerConfig(**config_dict)
