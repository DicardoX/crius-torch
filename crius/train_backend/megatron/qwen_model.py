#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Optional, Union
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import mpu
from megatron.core.transformer.enums import ModelType

from crius.train_backend.megatron.layers import ColumnParallelLinear, LanguageModelEmbedding
from crius.train_backend.megatron.llama_model import RMSNorm, LlamaTransformerLayer, LlamaTransformerBlock
from crius.train_backend.megatron.pipeline_parallel import schedules, p2p_communication
from crius.global_envs import logger


__all__ = ["QwenModel"]


class QwenModel(LanguageModule):

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if hasattr(config, 'gated_linear_unit') and not config.gated_linear_unit:
            warnings.warn(
                "Qwen models use SwiGLU activation (gated_linear_unit=True). "
                "Setting gated_linear_unit=False will not match Qwen architecture.",
                UserWarning
            )

        super().__init__(config)

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

        self.decoder = LlamaTransformerBlock(
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

        logger.info(f"Initialized QwenModel with vocab_size={vocab_size}, "
                   f"max_seq_length={max_sequence_length}, rope_theta={rope_theta}")

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        if input_tensor is None:
            return
        
        input_tensor_list = [input_tensor] if not isinstance(input_tensor, list) else input_tensor

        assert len(input_tensor_list) == 1, 'input_tensor should only be length 1 for Qwen'
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
            # 中间/最后阶段：decoder_input 将从 set_input_tensor 设置
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



def qwen2_5_0_5b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 24,
        'hidden_size': 896,
        'num_attention_heads': 14,
        'num_query_groups': 2,
        'ffn_hidden_size': 4864,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)


def qwen2_5_1_5b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 28,
        'hidden_size': 1536,
        'num_attention_heads': 12,
        'num_query_groups': 2,
        'ffn_hidden_size': 8960,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)


def qwen2_5_7b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 28,
        'hidden_size': 3584,
        'num_attention_heads': 28,
        'num_query_groups': 4,
        'ffn_hidden_size': 18944,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)


def qwen2_5_14b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 48,
        'hidden_size': 5120,
        'num_attention_heads': 40,
        'num_query_groups': 8,
        'ffn_hidden_size': 13824,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)


def qwen2_5_32b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 64,
        'hidden_size': 5120,
        'num_attention_heads': 40,
        'num_query_groups': 8,
        'ffn_hidden_size': 27648,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)


def qwen2_5_72b_config(**kwargs) -> TransformerConfig:
    config_dict = {
        'num_layers': 80,
        'hidden_size': 8192,
        'num_attention_heads': 64,
        'num_query_groups': 8,
        'ffn_hidden_size': 29568,
        'kv_channels': 128,
        'gated_linear_unit': True,
        'activation_func': F.silu,
        'add_bias_linear': False,
        'layernorm_epsilon': 1e-6,
    }
    config_dict.update(kwargs)
    return TransformerConfig(**config_dict)

