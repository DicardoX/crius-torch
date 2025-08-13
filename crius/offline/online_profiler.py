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
    op_type=node.layer_type
    batch_size=input_shapes[0][0]
    
    # Reset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if op_type == 'softmax':
        num_heads = input_shapes[0][1]
        seq_len = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_softmax(batch_size, num_heads, seq_len)
    elif op_type == 'layernorm':
        seq_len = input_shapes[0][1]
        embedding_dim = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_layernorm(batch_size, embedding_dim, seq_len)
    elif op_type == 'gelu':
        seq_len = input_shapes[0][1]
        embedding_dim = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_gelu(batch_size, seq_len, embedding_dim)
    elif op_type == 'dropout':
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_dropout(input_shapes[0])
    elif op_type == 'pow':
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_pow(input_shapes[0])
    elif op_type == 'embedding':
        vocab_size = input_shapes[0][0]
        embedding_dim = input_shapes[0][1]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_embedding(vocab_size, embedding_dim)
    elif op_type == 'sdpa':
        num_heads = input_shapes[0][1]
        seq_len = input_shapes[0][2]
        dim_per_head = input_shapes[0][3]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_sdpa(batch_size, num_heads, seq_len, dim_per_head)
    elif op_type == 'masked_fill':
        num_heads = input_shapes[0][1]
        seq_len = input_shapes[0][2]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_masked_fill(batch_size, num_heads, seq_len)
    elif op_type == 'contiguous':
        dim_per_head = None
        if len(input_shapes[0]) < 4:
            seq_len = input_shapes[0][1]
            num_heads_or_hidden_dim = input_shapes[0][2]
        else:
            num_heads_or_hidden_dim = input_shapes[0][1]
            seq_len = input_shapes[0][2]
            dim_per_head = input_shapes[0][3]
        
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_transpose_contiguous_view(batch_size, seq_len=seq_len, 
                                                                          num_heads_or_hidden_dim=num_heads_or_hidden_dim, dim_per_head=dim_per_head)
    elif op_type == 'addmm':
        # addmm 算子的输入输出信息不太清楚具体的含义，所以这里暂时把第一维信息放在batch_size中
        B = input_shapes[0][0]
        input_dim = input_shapes[1][1]
        output_dim = input_shapes[1][0]
        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.benchmark_addmm(B, input_dim, output_dim)
    else:
        seq_len=input_shapes[0][1]
        left_hidden_dim=input_shapes[0][2]
        right_hidden_dim=output_shapes[0][2]
        gemm_op_type=get_optype_func(op_type)

        query_key = CriusOfflineDataManager.get_query_key(op_type, input_shapes=input_shapes, output_shapes=output_shapes)
        op_latency = offline_profiler.bencmark_one_op(gemm_op_type, batch_size, seq_len, left_hidden_dim, right_hidden_dim)

    offline_profiler._write_one_entry(query_key, op_latency, None)
    offline_profiler._store()
    return op_latency
    

        