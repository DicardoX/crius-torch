#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Optional

import torch
from torch.nn.parameter import Parameter
from megatron.core import mpu

from crius.global_envs import logger

__all__ = [
    "ColumnParallelLinear",
]


class ColumnParallelLinear(torch.nn.Module):
    """ Clean version of Megatron's ColumnParallelLinear. """

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
    ):
        """ Forward of ColumnParallelLinear. """

        bias = self.bias if not self.skip_bias_add else None

        # TODO(chunyu): Add the autograd function for copy_to_tensor_model_parallel_region here.

        if not self.weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_, self.weight, bias)
        
        # TODO(chunyu): Add the autograd function for ather_from_tensor_model_parallel_region here.
        output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """ Clean version of Megatron's RowParallelLinear. """

    def __init__(self, input_size, output_size, tp_size, bias, device, dtype, skip_bias_add=False, **kwargs):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.input_size_per_partition = input_size // tp_size
        self.device = device
        self.dtype = dtype
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
    
    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        """ Forward of RowParallelLinear. """

        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            pass    # Skip shape check

        # TODO(chunyu): Add the autograd function for scatter_to_tensor_model_parallel_region() here.
        input_parallel = input_

        if not weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_parallel, weight, bias=None)
        
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
    """ Clean version of Megatron's linear_with_grad_accumulation_and_async_allreduce(). """    

    # TODO(chunyu): Currently we don't use customized autograd function.

    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output
