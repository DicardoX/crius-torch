#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""测试 LlamaModel 的 Crius Compile 流程

运行方式：
    python scripts/test_llama_compile.py
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ============== 辅助函数 ==============

def init_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        print(f"[Rank {rank}/{world_size}] 分布式环境初始化成功")
        return rank, world_size, local_rank
    else:
        print("单GPU模式")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def init_megatron_parallel():
    """初始化 Megatron 并行状态"""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    
    if not dist.is_initialized():
        print("初始化单GPU分布式环境...")
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23459', rank=0, world_size=1)
    
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
        print("✓ Megatron 并行状态初始化成功")
    except Exception as e:
        print(f"并行状态已初始化或初始化失败: {e}")


# ============== 测试用例 ==============

def test_compile_step1_transform_to_ir_graph():
    """测试 Step 1: transform_to_ir_graph - 使用 dynamo.export 将模型转为 IR Graph"""
    print("\n" + "=" * 70)
    print("测试 Step 1: transform_to_ir_graph (LlamaModel)")
    print("=" * 70)
    
    try:
        from crius.compile.graph.transform import transform_to_ir_graph
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
        
        # 初始化 Megatron 并行状态
        init_megatron_parallel()
        
        # 创建小配置的 LlamaModel
        config = llama2_7b_config(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,  # GQA
            ffn_hidden_size=1024,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        vocab_size = 1000
        max_seq_len = 128
        
        print(f"创建 LlamaModel:")
        print(f"  - num_layers: {config.num_layers}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - num_attention_heads: {config.num_attention_heads}")
        print(f"  - num_query_groups: {config.num_query_groups}")
        
        model = LlamaModel(
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_seq_len,
            rope_theta=10000.0,
        ).cuda()
        
        print("✓ LlamaModel 创建成功")
        
        # 创建输入数据
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, vocab_size, (batch_size, seq_len)).cuda()
        
        print(f"\n输入数据:")
        print(f"  - input_ids shape: {input_ids.shape}")
        print(f"  - position_ids shape: {position_ids.shape}")
        print(f"  - attention_mask shape: {attention_mask.shape}")
        
        # 尝试 transform_to_ir_graph
        print("\n调用 transform_to_ir_graph (dynamo.export)...")
        
        input_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        graph = transform_to_ir_graph(model, **input_kwargs)
        
        print(f"✓ transform_to_ir_graph 成功!")
        print(f"  - IR Graph 节点数: {len(list(graph.nodes))}")  # type: ignore
        
        # 打印前10个节点
        print(f"\n  前10个 IR Graph 节点:")
        nodes_list = list(graph.nodes)  # type: ignore
        for i, node in enumerate(nodes_list):
            if i >= 10:
                print(f"    ... (还有 {len(nodes_list) - 10} 个节点)")
                break
            print(f"    [{i}] op={node.op}, target={str(node.target)[:50]}")
        
        return True, graph
        
    except Exception as e:
        import traceback
        print(f"✗ transform_to_ir_graph 失败: {e}")
        traceback.print_exc()
        return False, None


def test_compile_step2_propagate_shapes(graph):
    """测试 Step 2: propagate_node_shapes - 推断每个节点的 shape"""
    print("\n" + "=" * 70)
    print("测试 Step 2: propagate_node_shapes")
    print("=" * 70)
    
    if graph is None:
        print("跳过: 没有 IR Graph")
        return False, None
    
    try:
        from crius.compile.graph.transform import propagate_node_shapes
        
        # 创建输入数据（与 Step 1 相同）
        vocab_size = 1000
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, vocab_size, (batch_size, seq_len)).cuda()
        
        input_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        print("调用 propagate_node_shapes...")
        graph = propagate_node_shapes(graph, **input_kwargs)
        
        print(f"✓ propagate_node_shapes 成功!")
        
        # 打印一些节点的 shape 信息
        print(f"\n  部分节点的 shape 信息:")
        for i, node in enumerate(graph.nodes):  # type: ignore
            if i >= 5:
                break
            input_shapes = getattr(node, 'input_shapes', [])
            output_shapes = getattr(node, 'output_shapes', [])
            print(f"    [{i}] {str(node.target)[:40]}")
            print(f"        input_shapes: {input_shapes}")
            print(f"        output_shapes: {output_shapes}")
        
        return True, graph
        
    except Exception as e:
        import traceback
        print(f"✗ propagate_node_shapes 失败: {e}")
        traceback.print_exc()
        return False, None


def test_compile_step3_infer_layer_types(graph):
    """测试 Step 3: infer_node_layer_types - 推断节点类型"""
    print("\n" + "=" * 70)
    print("测试 Step 3: infer_node_layer_types")
    print("=" * 70)
    
    if graph is None:
        print("跳过: 没有 IR Graph")
        return False, None
    
    try:
        from crius.compile.graph.transform import infer_node_layer_types
        
        print("调用 infer_node_layer_types...")
        graph = infer_node_layer_types(graph)
        
        print(f"✓ infer_node_layer_types 成功!")
        
        # 统计各类型节点数量
        layer_type_counts = {}
        for node in graph.nodes:  # type: ignore
            layer_type = getattr(node, 'layer_type', 'unknown')
            layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1
        
        print(f"\n  节点类型统计:")
        for layer_type, count in sorted(layer_type_counts.items()):
            print(f"    {layer_type}: {count}")
        
        return True, graph
        
    except Exception as e:
        import traceback
        print(f"✗ infer_node_layer_types 失败: {e}")
        traceback.print_exc()
        return False, None


# ============== 主函数 ==============

def main():
    """主测试函数"""
    print("=" * 70)
    print("LlamaModel Compile 流程测试")
    print("=" * 70)
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()
    
    results = {}
    
    # 测试 Step 1: transform_to_ir_graph
    success, graph = test_compile_step1_transform_to_ir_graph()
    results['Step1: transform_to_ir_graph'] = success
    
    # 测试 Step 2: propagate_node_shapes
    if success:
        success, graph = test_compile_step2_propagate_shapes(graph)
        results['Step2: propagate_node_shapes'] = success
    
    # 测试 Step 3: infer_node_layer_types
    if success:
        success, graph = test_compile_step3_infer_layer_types(graph)
        results['Step3: infer_node_layer_types'] = success
    
    # 打印结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:40s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有 Compile 流程测试通过！")
    else:
        print("❌ 部分 Compile 流程测试失败！")
    print("=" * 70)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
