import os
import torch
import gc

from typing import (Any, Dict, List, Optional, Tuple, Union, Callable)
from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
from crius.compile.graph import IRNode
from crius.global_envs import logger


def get_optype_func(op_type: str) -> Callable:
    if op_type == 'add':
        return torch.add
    elif op_type == 'mul':
        return torch.mul
    elif op_type == 'matmul':
        return torch.matmul
    elif op_type == 'eq':
        return torch.eq
    elif op_type == 'tril':
        return torch.tril
    elif op_type == 'gemm':
        return torch.matmul
    elif op_type == 'linear':
        return torch.nn.Linear
    else:
        return None

def online_single_op(node: IRNode) -> float:

    profile_configs = OfflineProfileConfigs()
    offline_profiler = CriusOfflineDataManager(
        profile_configs=profile_configs,
        to_store_path='./crius/offline/',
        data_filename='prof_data.pkl',
    )

    input_shapes, output_shapes = node.input_shapes, node.output_shapes
    op_type = node.layer_type
    dtype = node.dtype
    batch_size = input_shapes[0][0]
    
    # Reset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    ifWrite = True

    if op_type == 'softmax':
        if len(input_shapes[0]) >= 3:
            num_heads = input_shapes[0][1]
            seq_len = input_shapes[0][2]
            hidden_size = 0
        elif len(input_shapes[0]) >= 2:
            seq_len = input_shapes[0][0]
            hidden_size = input_shapes[0][1]
            batch_size = 0
            num_heads = 0
        
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_softmax(batch_size, num_heads, seq_len, hidden_size, dtype=dtype)
    elif op_type == 'layernorm':
        seq_len = input_shapes[0][1]
        embedding_dim = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_layernorm(batch_size, embedding_dim, seq_len, dtype=dtype)
    elif op_type == 'gelu':
        seq_len = input_shapes[0][1]
        embedding_dim = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_gelu(batch_size, seq_len, embedding_dim, dtype=dtype)
    elif op_type == 'expert_gelu':
        top_k = input_shapes[0][0]
        embedding_dim = input_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_expert_gelu(top_k, embedding_dim, dtype=dtype)
    elif op_type == 'dropout':
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_dropout(input_shapes[0], dtype=dtype)
    elif op_type == 'pow' and node.op == 'call_function':
        op_type = 'function_pow'
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_pow(input_shapes[0], 'function', dtype=dtype)
    elif op_type == 'pow' and node.op == 'call_method':
        op_type = 'method_pow'
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_pow(input_shapes[0], 'method', dtype=dtype)
    elif op_type == 'embedding':
        vocab_size = input_shapes[0][0]
        embedding_dim = input_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_embedding(vocab_size, embedding_dim, dtype=dtype)
    elif op_type == 'sdpa':
        num_heads = input_shapes[0][1]
        seq_len = input_shapes[0][2]
        dim_per_head = input_shapes[0][3]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_sdpa(batch_size, num_heads, seq_len, dim_per_head, dtype=dtype)
    elif op_type == 'masked_fill':
        num_heads = input_shapes[0][1]
        seq_len = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_masked_fill(batch_size, num_heads, seq_len, dtype=dtype)
    elif op_type == 'contiguous':
        dim_per_head = None
        if len(input_shapes[0]) < 4:
            seq_len = input_shapes[0][1]
            num_heads_or_hidden_dim = input_shapes[0][2]
        else:
            num_heads_or_hidden_dim = input_shapes[0][1]
            seq_len = input_shapes[0][2]
            dim_per_head = input_shapes[0][3]
        
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_transpose_contiguous_view(batch_size, seq_len=seq_len, 
                                                                          num_heads_or_hidden_dim=num_heads_or_hidden_dim, dim_per_head=dim_per_head,
                                                                          dtype=dtype)
    elif op_type == 'addmm':
        # addmm 算子的输入输出信息不太清楚具体的含义，所以这里暂时把第一维信息放在batch_size中
        B = input_shapes[0][0]
        input_dim = input_shapes[1][1]
        output_dim = input_shapes[1][0]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_addmm(B, input_dim, output_dim, dtype=dtype)
    elif op_type == 'tril':
        seq_len = output_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_tril(seq_len, dtype=dtype)
    elif op_type == 'topk':
        seq_len = input_shapes[0][0]
        expert_nums = input_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_topk(bs=1, topk=8, expert_nums=expert_nums, seq_len=seq_len, dtype=dtype)
    elif op_type == 'rsqrt':
        ifWrite = False
        op_latency = offline_profiler.benchmark_rsqrt(input_shapes, dtype=dtype)
    elif op_type == 'to':
        ifWrite = False
        op_latency = offline_profiler.benchmark_to_dtype(input_shapes, output_shapes, dtype=dtype)
    elif op_type == 'linear':
        seq_len=input_shapes[0][1]
        left_hidden_dim=input_shapes[0][2]
        right_hidden_dim=output_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_linear(batch_size, seq_len, left_hidden_dim, right_hidden_dim, dtype=dtype)
    elif op_type == 'expert_linear':
        top_k = input_shapes[0][0]
        left_hidden_dim = input_shapes[0][1]
        right_hidden_dim = output_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_expert_linear(top_k, left_hidden_dim, right_hidden_dim, dtype=dtype)
    else:

        if len(input_shapes[0]) >= 3:
            batch_size = input_shapes[0][0]               
            seq_len=input_shapes[0][1]
            left_hidden_dim=input_shapes[0][2]
            right_hidden_dim=output_shapes[0][2]
        elif len(input_shapes[1]) >= 3:
            batch_size = input_shapes[1][0]               
            seq_len=input_shapes[1][1]
            left_hidden_dim=input_shapes[1][2]
            right_hidden_dim=output_shapes[0][2]
        else: 
            # in expert layer, mul always like: 
            # input_shapes = [torch.Size([4, 768]), torch.Size([4, 768])], output_shapes = [torch.Size([4, 768])]
            # the 1st is top_k, the 2nd num is hidden dim
            top_k = input_shapes[0][0]
            left_hidden_dim = input_shapes[0][1]
            right_hidden_dim = input_shapes[1][1]
            batch_size = top_k
            seq_len = 0
            
        gemm_op_type=get_optype_func(op_type)

        query_key = CriusOfflineDataManager.get_query_key(op_type, dtype, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.bencmark_one_op(gemm_op_type, batch_size, seq_len, left_hidden_dim, right_hidden_dim, dtype=dtype)

    if ifWrite == True:
        offline_profiler._write_one_entry(query_key, op_latency, None)
        offline_profiler._store()
    return op_latency
    

        