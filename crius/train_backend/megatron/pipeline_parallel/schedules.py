#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""流水线并行调度模块 - 基于 Megatron-LM 官方实现简化"""

from typing import Callable, Optional
import torch
from torch import Tensor

from megatron.core import parallel_state
from . import p2p_communication


def get_forward_backward_func():
    """根据流水线并行大小选择合适的调度函数"""
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    
    if pipeline_model_parallel_size > 1:
        # 使用流水线并行
        return forward_backward_pipelining_without_interleaving
    else:
        # 无流水线并行
        return forward_backward_no_pipelining


def forward_step(
    model,
    data_iterator,
    input_tensor: Optional[Tensor],
    forward_only: bool = False,
):
    """执行单个前向步骤
    
    Args:
        model: 模型
        data_iterator: 数据迭代器（只有第一阶段需要）
        input_tensor: 从上一阶段接收的输入张量
        forward_only: 是否只做前向（推理模式）
        
    Returns:
        (output_tensor, loss_func): 输出张量和损失函数
    """
    # 只有第一个流水线阶段需要从data_iterator获取数据
    if parallel_state.is_pipeline_first_stage():
        if data_iterator is not None:
            try:
                data = next(data_iterator)
            except StopIteration:
                data = None
            
            if data is None:
                return None, None
        else:
            raise RuntimeError("First pipeline stage requires data_iterator")
    else:
        # 中间和最后阶段创建占位符数据
        if input_tensor is not None:
            batch_size = input_tensor.size(1)
            seq_len = input_tensor.size(0)
            device = input_tensor.device
        else:
            batch_size = 1
            seq_len = 1
            device = torch.cuda.current_device()
        
        data = {
            'input_ids': torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),
            'position_ids': torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),
            'attention_mask': torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device),
        }
    
    # 设置输入张量
    if input_tensor is not None:
        model.set_input_tensor(input_tensor)
    
    # 前向传播
    with torch.set_grad_enabled(not forward_only):
        output_tensor = model(**data)
    
    return output_tensor, None


def backward_step(
    optimizer,
    input_tensor: Optional[Tensor],
    output_tensor: Optional[Tensor],
    output_tensor_grad: Optional[Tensor],
):
    """执行单个反向步骤
    
    Args:
        optimizer: 优化器
        input_tensor: 输入张量
        output_tensor: 输出张量
        output_tensor_grad: 输出梯度
        
    Returns:
        input_tensor_grad: 输入梯度
    """
    # 最后一个阶段的特殊处理
    if parallel_state.is_pipeline_last_stage():
        if output_tensor is not None:
            # 标量loss直接backward
            if output_tensor.numel() == 1:
                output_tensor.backward()
            else:
                # 非标量，使用全1梯度
                grad = torch.ones_like(output_tensor)
                torch.autograd.backward(output_tensor, grad_tensors=grad)
    else:
        # 中间阶段的反向传播
        if output_tensor is not None and output_tensor_grad is not None:
            torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    
    # 获取输入梯度（所有阶段都需要，包括last stage）
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    
    return input_tensor_grad


def forward_backward_no_pipelining(
    forward_step_func: Callable,
    data_iterator,
    model,
    optimizer,
    num_microbatches: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """无流水线并行的前向后向传播"""
    
    no_sync_func = None
    if hasattr(model, 'no_sync'):
        no_sync_func = model.no_sync
    
    losses = []
    forward_data_store = []
    
    # 前向传播所有微批次
    for i in range(num_microbatches):
        output_tensor, loss_func = forward_step_func(
            model=model,
            data_iterator=data_iterator,
            input_tensor=None,
            forward_only=forward_only,
        )
        
        if parallel_state.is_pipeline_last_stage():
            if output_tensor is not None:
                losses.append(output_tensor)
        
        forward_data_store.append((output_tensor, loss_func))
    
    # 反向传播所有微批次
    if not forward_only:
        for i in range(num_microbatches):
            output_tensor, loss_func = forward_data_store[i]
            
            # 最后一个微批次之前禁用梯度同步
            if no_sync_func is not None and i < num_microbatches - 1:
                with no_sync_func():
                    backward_step(
                        optimizer=optimizer,
                        input_tensor=None,
                        output_tensor=output_tensor,
                        output_tensor_grad=None,
                    )
            else:
                backward_step(
                    optimizer=optimizer,
                    input_tensor=None,
                    output_tensor=output_tensor,
                    output_tensor_grad=None,
                )
    
    return losses


def forward_backward_pipelining_without_interleaving(
    forward_step_func: Callable,
    data_iterator,
    model,
    optimizer,
    num_microbatches: int,
    seq_len: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """1F1B流水线并行调度（无交错）
    
    基于 Megatron-LM 官方实现，包含三个阶段：
    1. Warmup: 填充流水线（只做前向）
    2. Steady State: 1F1B模式（一个前向，一个反向）
    3. Cooldown: 清空流水线（只做反向）
    """
    
    # 计算warmup微批次数量
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    
    num_warmup_microbatches = (
        pipeline_parallel_size - pipeline_parallel_rank - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    
    # 获取张量形状和数据类型
    def get_tensor_shape():
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            # [seq_len, batch_size, hidden_size]
            return [seq_len, num_microbatches, model.config.hidden_size]
        return None
    
    def get_dtype():
        if hasattr(model, 'config'):
            return getattr(model.config, 'pipeline_dtype', torch.float32)
        return torch.float32
    
    tensor_shape = get_tensor_shape()
    dtype = get_dtype()
    
    # 用于保存中间结果
    input_tensors = []
    output_tensors = []
    losses = []
    
    # ========== Warmup 阶段 ==========
    # 填充流水线，只做前向传播
    for i in range(num_warmup_microbatches):
        # 1. 接收输入（第一阶段返回None）
        input_tensor = p2p_communication.recv_forward(
            tensor_shape=tensor_shape,
            dtype=dtype,
        )
        
        # 2. 前向传播
        output_tensor, _ = forward_step_func(
            model=model,
            data_iterator=data_iterator,
            input_tensor=input_tensor,
            forward_only=forward_only,
        )
        
        # 3. 发送输出（最后阶段不发送）
        p2p_communication.send_forward(output_tensor=output_tensor)
        
        # 4. 保存以供反向传播使用
        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        # 5. 收集损失
        if parallel_state.is_pipeline_last_stage() and output_tensor is not None:
            losses.append(output_tensor)
    
    # ========== Steady State 阶段 ==========
    # 1F1B模式：前向一个新的，反向一个旧的
    
    # 在进入1F1B循环前，需要预先接收第一个输入
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(
            tensor_shape=tensor_shape,
            dtype=dtype,
        )
    
    for i in range(num_microbatches_remaining):
        last_iteration = (i == num_microbatches_remaining - 1)
        
        # 1. 前向传播
        output_tensor, _ = forward_step_func(
            model=model,
            data_iterator=data_iterator,
            input_tensor=input_tensor,
            forward_only=forward_only,
        )
        
        # 收集损失
        if parallel_state.is_pipeline_last_stage() and output_tensor is not None:
            losses.append(output_tensor)
        
        if forward_only:
            # 推理模式：只做前向
            p2p_communication.send_forward(output_tensor=output_tensor)
            
            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                )
        else:
            # 训练模式：1F1B
            
            # 2. 发送前向输出，同时接收反向梯度
            output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor=output_tensor,
                tensor_shape=tensor_shape,
                dtype=dtype,
            )
            
            # 3. 保存当前的input和output到列表末尾
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            
            # 4. 取出最早的input和output做反向传播
            input_tensor_to_backward = input_tensors.pop(0)
            output_tensor_to_backward = output_tensors.pop(0)
            
            # 5. 反向传播
            input_tensor_grad = backward_step(
                optimizer=optimizer,
                input_tensor=input_tensor_to_backward,
                output_tensor=output_tensor_to_backward,
                output_tensor_grad=output_tensor_grad,
            )
            
            # 6. 发送反向梯度，同时接收下一个前向输入
            if last_iteration:
                # 最后一次迭代：只发送梯度
                p2p_communication.send_backward(input_tensor_grad=input_tensor_grad)
                input_tensor = None
            else:
                # 发送梯度，同时接收下一个输入
                input_tensor = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad=input_tensor_grad,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                )
    
    # ========== Cooldown 阶段 ==========
    # 清空流水线，只做反向传播
    if not forward_only:
        for i in range(num_warmup_microbatches):
            # 1. 接收梯度（最后阶段返回None）
            output_tensor_grad = p2p_communication.recv_backward(
                tensor_shape=tensor_shape,
                dtype=dtype,
            )
            
            # 2. 取出保存的input和output
            input_tensor_to_backward = input_tensors.pop(0)
            output_tensor_to_backward = output_tensors.pop(0)
            
            # 3. 反向传播
            input_tensor_grad = backward_step(
                optimizer=optimizer,
                input_tensor=input_tensor_to_backward,
                output_tensor=output_tensor_to_backward,
                output_tensor_grad=output_tensor_grad,
            )
            
            # 4. 发送梯度（第一阶段不发送）
            p2p_communication.send_backward(input_tensor_grad=input_tensor_grad)
    
    return losses


# 交错式流水线并行的别名（暂时使用非交错实现）
def forward_backward_pipelining_with_interleaving(
    forward_step_func: Callable,
    data_iterator,
    model,
    optimizer,
    num_microbatches: int,
    seq_len: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """交错式流水线并行（暂时使用非交错实现）"""
    # 如果model不是list，使用非交错版本
    if not isinstance(model, list):
        return forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            optimizer=optimizer,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
        )
    
    # TODO: 实现真正的交错式流水线并行
    raise NotImplementedError("Interleaved pipeline parallelism with model chunking not implemented yet")

