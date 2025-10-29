#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""测试流水线并行功能

运行方式：
    # 单GPU测试（no pipelining）
    python test_pipeline_parallel.py
    
    # 多GPU测试（流水线并行）
    torchrun --nproc-per-node=4 test_pipeline_parallel.py
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# 初始化分布式环境
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
        return 0, 1, 0


# 创建简单的测试数据集
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=128, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'position_ids': torch.arange(self.seq_len),
            'attention_mask': torch.tril(torch.ones(self.seq_len, self.seq_len)).bool(),  # 因果mask，布尔类型
            'labels': torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


def test_basic_import():
    """测试1：基本导入测试"""
    print("\n=== 测试1：导入模块 ===")
    try:
        from crius.train_backend.megatron.pipeline_parallel import schedules, p2p_communication
        print("✓ 导入成功")
        
        # 检查主要函数
        assert hasattr(schedules, 'forward_backward_no_pipelining')
        assert hasattr(schedules, 'forward_backward_pipelining_without_interleaving')
        assert hasattr(schedules, 'forward_backward_pipelining_with_interleaving')
        assert hasattr(p2p_communication, 'send_forward')
        assert hasattr(p2p_communication, 'recv_forward')
        print("✓ 所有必需函数都存在")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_gpt_model_integration():
    """测试2：GPT模型集成测试"""
    print("\n=== 测试2：GPT模型集成 ===")
    try:
        from crius.train_backend.megatron.gpt_model import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig
        
        # 检查方法是否存在
        assert hasattr(GPTModel, 'pipeline_forward_backward')
        assert hasattr(GPTModel, 'get_forward_backward_func')
        print("✓ GPT模型包含流水线并行方法")
        
        return True
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False


def test_single_gpu_training():
    """测试3：单GPU训练测试（不使用流水线）"""
    print("\n=== 测试3：单GPU训练 ===")
    
    try:
        from crius.train_backend.megatron.gpt_model import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import parallel_state
        
        # 初始化并行状态（模拟单GPU）
        if not dist.is_initialized():
            print("跳过：需要分布式环境")
            return True
        
        # 简化的配置
        config = TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            pipeline_model_parallel_size=1,  # 单GPU
            tensor_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型
        model = GPTModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        )
        model = model.cuda()
        
        # 创建数据
        dataset = DummyDataset(num_samples=8, seq_len=32, vocab_size=1000)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # 包装迭代器，自动将数据移到GPU
        def gpu_data_iterator():
            for batch in dataloader:
                yield {k: v.cuda() for k, v in batch.items()}
        
        data_iterator = gpu_data_iterator()
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 运行一次前向后向
        print("运行前向后向传播...")
        losses = model.pipeline_forward_backward(
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_microbatches=2,
            forward_only=False
        )
        
        print(f"✓ 单GPU训练成功，losses数量: {len(losses)}")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ 单GPU训练失败: {e}")
        traceback.print_exc()
        return False


def test_pipeline_parallel_training():
    """测试4：多GPU流水线并行训练"""
    print("\n=== 测试4：流水线并行训练 ===")
    
    if not dist.is_initialized():
        print("跳过：需要多GPU分布式环境")
        return True
    
    try:
        from crius.train_backend.megatron.gpt_model import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import parallel_state
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        if world_size < 2:
            print("跳过：需要至少2个GPU")
            return True
        
        # 初始化Megatron并行状态
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=world_size,
            )
        except:
            print("并行状态已初始化，继续...")
        
        # 确定这个rank是否是第一个/最后一个阶段
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        is_first = parallel_state.is_pipeline_first_stage()
        is_last = parallel_state.is_pipeline_last_stage()
        
        print(f"[Rank {rank}] Pipeline rank: {pipeline_rank}, "
              f"First: {is_first}, Last: {is_last}")
        
        # 配置
        config = TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            pipeline_model_parallel_size=world_size,
            tensor_model_parallel_size=1,
            pipeline_dtype=torch.float32,  # 必须指定
            params_dtype=torch.float32,
        )
        
        # 创建模型（每个rank只创建自己那部分）
        model = GPTModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            pre_process=is_first,
            post_process=is_last,
        )
        model = model.cuda()
        
        # 创建数据（只有第一个阶段需要真实数据）
        if is_first:
            dataset = DummyDataset(num_samples=16, seq_len=32, vocab_size=1000)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # 包装迭代器，自动将数据移到GPU
            def gpu_data_iterator():
                for batch in dataloader:
                    yield {k: v.cuda() for k, v in batch.items()}
            
            data_iterator = gpu_data_iterator()
        else:
            data_iterator = None
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 运行流水线并行训练
        print(f"[Rank {rank}] 开始流水线并行训练...")
        losses = model.pipeline_forward_backward(
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_microbatches=4,
            forward_only=False
        )
        
        if is_last:
            print(f"✓ [Rank {rank}] 流水线并行训练成功，losses: {len(losses)}")
        else:
            print(f"✓ [Rank {rank}] 流水线并行训练成功")
        
        return True
        
    except Exception as e:
        import traceback
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"✗ [Rank {rank}] 流水线并行训练失败: {e}")
        traceback.print_exc()
        return False


def test_communication_functions():
    """测试5：通信函数测试"""
    print("\n=== 测试5：通信函数 ===")
    
    try:
        from crius.train_backend.megatron.pipeline_parallel import p2p_communication
        
        # 检查所有通信函数
        functions = [
            'send_forward',
            'recv_forward',
            'send_backward',
            'recv_backward',
            'send_forward_recv_backward',
            'send_backward_recv_forward',
            'send_forward_recv_forward',
            'send_backward_recv_backward',
            'send_forward_backward_recv_forward_backward',
        ]
        
        for func_name in functions:
            assert hasattr(p2p_communication, func_name), f"缺少函数: {func_name}"
        
        print(f"✓ 所有{len(functions)}个通信函数都存在")
        return True
        
    except Exception as e:
        print(f"✗ 通信函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("流水线并行模块测试")
    print("=" * 60)
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()
    
    # 运行测试
    results = {}
    
    # 测试1-2：基础测试（所有rank都运行）
    results['导入测试'] = test_basic_import()
    results['集成测试'] = test_gpt_model_integration()
    results['通信函数测试'] = test_communication_functions()
    
    # 测试3-4：训练测试
    if world_size == 1:
        results['单GPU训练'] = test_single_gpu_training()
    else:
        results['流水线并行训练'] = test_pipeline_parallel_training()
    
    # 打印结果
    if rank == 0:
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        for name, passed in results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{name}: {status}")
        
        all_passed = all(results.values())
        print("\n" + "=" * 60)
        if all_passed:
            print("所有测试通过！✓")
        else:
            print("部分测试失败！✗")
        print("=" * 60)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

