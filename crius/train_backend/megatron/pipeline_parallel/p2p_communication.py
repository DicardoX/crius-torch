#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""点对点通信模块 - 基于 Megatron-LM 官方实现简化"""

from typing import List, Optional, Tuple
import torch
from megatron.core import parallel_state


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
):
    """批量执行P2P操作"""
    prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_rank, group
        )
        ops.append(recv_next_op)
    
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        # 保护避免竞态条件
        torch.cuda.synchronize()


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: List[int],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """核心通信函数
    
    Args:
        tensor_send_next: 发送到下一阶段的张量
        tensor_send_prev: 发送到上一阶段的张量  
        recv_prev: 是否从上一阶段接收
        recv_next: 是否从下一阶段接收
        tensor_shape: 接收张量的形状
        dtype: 数据类型
        
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    # 创建接收缓冲区
    tensor_recv_prev = None
    tensor_recv_next = None
    
    if recv_prev:
        if tensor_shape is None:
            raise RuntimeError("tensor_shape must be specified if recv_prev is True")
        tensor_recv_prev = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    
    if recv_next:
        if tensor_shape is None:
            raise RuntimeError("tensor_shape must be specified if recv_next is True")
        tensor_recv_next = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    
    # 执行批量P2P通信
    group = parallel_state.get_pipeline_model_parallel_group()
    _batched_p2p_ops(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=group,
    )
    
    return tensor_recv_prev, tensor_recv_next


def recv_forward(tensor_shape: List[int], dtype: torch.dtype):
    """接收前向激活值"""
    if parallel_state.is_pipeline_first_stage():
        return None
    
    tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return tensor


def recv_backward(tensor_shape: List[int], dtype: torch.dtype):
    """接收反向梯度"""
    if parallel_state.is_pipeline_last_stage():
        return None
    
    _, tensor = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return tensor


def send_forward(output_tensor: torch.Tensor):
    """发送前向激活值"""
    if parallel_state.is_pipeline_last_stage():
        return
    
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        dtype=None,
    )


def send_backward(input_tensor_grad: torch.Tensor):
    """发送反向梯度"""
    if parallel_state.is_pipeline_first_stage():
        return
    
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        dtype=None,
    )


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    tensor_shape: List[int],
    dtype: torch.dtype,
):
    """发送前向激活值，同时接收反向梯度"""
    if parallel_state.is_pipeline_last_stage():
        return None
    
    _, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: List[int],
    dtype: torch.dtype,
):
    """发送反向梯度，同时接收前向激活值"""
    if parallel_state.is_pipeline_first_stage():
        return None
    
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return input_tensor


# 以下函数为交错式流水线提供，暂时简化实现

def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: List[int],
    dtype: torch.dtype,
):
    """前向传播期间，同时发送和接收激活值"""
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: List[int],
    dtype: torch.dtype,
):
    """反向传播期间，同时发送和接收梯度"""
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: List[int],
    dtype: torch.dtype,
):
    """同时进行前向和反向的发送和接收"""
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype=dtype,
    )
    return input_tensor, output_tensor_grad

