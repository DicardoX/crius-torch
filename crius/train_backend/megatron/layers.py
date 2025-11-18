#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's parallel linear layers. """

import os
from typing import Literal, Optional

import torch
from torch.nn.parameter import Parameter
from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [
    "LanguageModelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
]


class LanguageModelEmbedding(MegatronModule):
    """ Compilable version of Megatron's LanguageModelEmbedding. """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
    ):
        super().__init__(config)

        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'

        # TODO(chunyu): Support Megatron's VocabParallelEmbedding.
        self.word_embeddings = torch.nn.Embedding(
            self.vocab_size,
            self.config.hidden_size,
        )
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length,
                self.config.hidden_size,
            )
        
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Forward of LanguageModelEmbedding. """

        # Word embeddings
        words_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class ColumnParallelLinear(torch.nn.Module):
    """ Compilable version of Megatron's ColumnParallelLinear. """

    def __init__(self, input_size, output_size, tp_size, bias, device, dtype, skip_bias_add=False, **kwargs):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.output_size_per_partition = output_size // tp_size
        self.device = device
        self.dtype = dtype
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=self.device,
                dtype=self.dtype,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """ Forward of ColumnParallelLinear. 
        
        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias
        """

        bias = self.bias if not self.skip_bias_add else None

        # TODO(chunyu): Add the autograd function for copy_to_tensor_model_parallel_region here.

        if not self.weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_, self.weight, bias)
        
        # TODO(chunyu): Add the autograd function for gather_from_tensor_model_parallel_region here.
        output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """ Compilable version of Megatron's RowParallelLinear. """

    def __init__(self, input_size, output_size, tp_size, bias, device, dtype, input_is_parallel=False, skip_bias_add=False, **kwargs):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.input_size_per_partition = input_size // tp_size
        self.device = device
        self.dtype = dtype
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=self.device,
                dtype=self.dtype,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.register_parameter("bias", None)
    
    def forward(self, input_: torch.Tensor):
        """ Forward of RowParallelLinear. """

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            # TODO(chunyu): Add the autograd function for scatter_to_tensor_model_parallel_region() here.
            # input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
            raise NotImplementedError
        
        # Matrix multiply.
        allreduce_dgrad = False

        if not self.weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_parallel, self.weight, bias=None)
        
        torch.distributed.all_reduce(output_parallel.contiguous(), group=mpu.get_tensor_model_parallel_group())
        output_ = output_parallel

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    """ Compilable version of Megatron's linear_with_grad_accumulation_and_async_allreduce(). """    

    # TODO(chunyu): Currently we don't use customized autograd function.

    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output
