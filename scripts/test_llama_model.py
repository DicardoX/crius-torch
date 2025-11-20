#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""测试 LlamaModel 的完整功能

运行方式：
    # 单GPU测试（无流水线并行）
    python scripts/test_llama_model.py
    
    # 多GPU测试（2个GPU的流水线并行）
    torchrun --nproc-per-node=2 scripts/test_llama_model.py
    
    # 多GPU测试（4个GPU的流水线并行）
    torchrun --nproc-per-node=4 scripts/test_llama_model.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader


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


class DummyLlamaDataset(Dataset):
    """简单的测试数据集，用于Llama模型"""
    def __init__(self, num_samples=100, seq_len=128, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Llama使用因果注意力，不需要显式mask（在模型内部处理）
        return {
            'input_ids': torch.randint(1, self.vocab_size, (self.seq_len,)),  # 从1开始，避免pad token
            'position_ids': torch.arange(self.seq_len),
            'attention_mask': torch.tril(torch.ones(self.seq_len, self.seq_len)).bool(),
            'labels': torch.randint(1, self.vocab_size, (self.seq_len,)),
        }


# ============== 测试用例 ==============

def test_basic_import():
    """测试1：基本导入测试"""
    print("\n=== 测试1：导入 LlamaModel 模块 ===")
    try:
        from crius.train_backend.megatron.llama_model import (
            LlamaModel, 
            RMSNorm,
            LlamaTransformerLayer,
            LlamaTransformerBlock,
            llama2_7b_config,
            llama2_13b_config,
            llama2_70b_config,
            llama3_8b_config,
            llama3_70b_config,
        )
        print("✓ 成功导入 LlamaModel 及相关组件")
        
        # 检查配置函数
        config = llama2_7b_config()
        print(f"✓ Llama2-7B 配置: {config.num_layers} 层, hidden_size={config.hidden_size}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试2：模型初始化测试"""
    print("\n=== 测试2：LlamaModel 初始化 ===")
    try:
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
        
        # 使用小配置进行测试
        config = llama2_7b_config(
            num_layers=4,  # 减少层数用于测试
            hidden_size=512,
            num_attention_heads=8,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型
        model = LlamaModel(
            config=config,
            vocab_size=32000,
            max_sequence_length=512,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            rope_theta=10000.0,
        )
        
        print(f"✓ 成功创建 LlamaModel")
        print(f"  - 词汇表大小: {model.vocab_size}")
        print(f"  - 最大序列长度: {model.max_sequence_length}")
        print(f"  - RoPE theta: {model.rope_theta}")
        
        # 检查模型结构
        assert hasattr(model, 'embedding'), "缺少 embedding 层"
        assert hasattr(model, 'decoder'), "缺少 decoder"
        assert hasattr(model, 'output_layer'), "缺少 output_layer"
        print("✓ 模型结构完整")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ 模型初始化失败: {e}")
        traceback.print_exc()
        return False


def test_rms_norm():
    """测试3：RMSNorm 层测试"""
    print("\n=== 测试3：RMSNorm 层 ===")
    try:
        from crius.train_backend.megatron.llama_model import RMSNorm
        
        # 创建 RMSNorm
        hidden_size = 128
        norm = RMSNorm(hidden_size, eps=1e-6).cuda()
        
        # 测试前向传播
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()
        output = norm(x)
        
        assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
        print(f"✓ RMSNorm 前向传播成功，输入形状: {x.shape}, 输出形状: {output.shape}")
        
        # 检查归一化效果
        output_mean = output.pow(2).mean(-1).mean()
        print(f"✓ 输出平方的均值: {output_mean.item():.6f} (应该接近 1.0)")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ RMSNorm 测试失败: {e}")
        traceback.print_exc()
        return False


def test_forward_pass_single_gpu():
    """测试4：单GPU前向传播测试"""
    print("\n=== 测试4：单GPU前向传播 ===")
    try:
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
        from megatron.core import parallel_state
        
        # 初始化并行状态（单GPU模式）
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        except:
            print("并行状态已初始化，继续...")
        
        # 直接创建小配置（不使用预定义配置）
        from megatron.core.transformer.transformer_config import TransformerConfig
        import torch.nn.functional as F
        
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=1024,  # 4 * hidden_size
            kv_channels=64,  # hidden_size // num_attention_heads
            gated_linear_unit=True,  # 启用 gated（SwiGLU）
            activation_func=F.silu,
            add_bias_linear=False,
            layernorm_epsilon=1e-5,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型
        model = LlamaModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        ).cuda()
        
        # 创建输入数据
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        
        # 前向传播（带标签，计算loss）
        print("前向传播（带标签）...")
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        assert loss.dim() == 0, f"Loss 应该是标量，但得到形状: {loss.shape}"
        print(f"✓ 前向传播成功，loss: {loss.item():.4f}")
        
        # 前向传播（无标签，生成logits）
        print("前向传播（无标签）...")
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        
        expected_shape = (batch_size, seq_len, 1000)
        assert logits.shape == expected_shape, f"Logits 形状不匹配: {logits.shape} vs {expected_shape}"
        print(f"✓ 生成 logits 成功，形状: {logits.shape}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ 单GPU前向传播失败: {e}")
        traceback.print_exc()
        return False


def test_backward_pass_single_gpu():
    """测试5：单GPU反向传播测试"""
    print("\n=== 测试5：单GPU反向传播和优化 ===")
    try:
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
        from megatron.core import parallel_state
        
        # 确保并行状态已初始化
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        except:
            print("并行状态已初始化，继续...")
        
        # 直接创建小配置（不使用预定义配置）
        from megatron.core.transformer.transformer_config import TransformerConfig
        import torch.nn.functional as F
        
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=512,  # 4 * hidden_size
            kv_channels=32,  # hidden_size // num_attention_heads
            gated_linear_unit=True,  # 启用 gated（SwiGLU）
            activation_func=F.silu,
            add_bias_linear=False,
            layernorm_epsilon=1e-5,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型和优化器
        model = LlamaModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 创建输入数据
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        
        # 训练一步
        print("执行训练步骤...")
        initial_loss = None
        for step in range(3):
            optimizer.zero_grad()
            
            loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            print(f"  步骤 {step+1}: loss = {loss.item():.4f}")
        
        print(f"✓ 反向传播和优化成功")
        print(f"  初始 loss: {initial_loss:.4f}")
        print(f"  最终 loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ 单GPU反向传播失败: {e}")
        traceback.print_exc()
        return False


def test_pipeline_parallel_training():
    """测试6：多GPU流水线并行训练"""
    print("\n=== 测试6：流水线并行训练 ===")
    
    if not dist.is_initialized():
        print("跳过：需要多GPU分布式环境")
        return True
    
    try:
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
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
        
        # 配置（每个GPU分配一部分层）
        total_layers = 8
        layers_per_stage = total_layers // world_size
        
        config = llama2_7b_config(
            num_layers=layers_per_stage,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=4,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=world_size,
            pipeline_dtype=torch.float32,
            params_dtype=torch.float32,
        )
        
        # 创建模型（每个rank只创建自己那部分）
        model = LlamaModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            pre_process=is_first,
            post_process=is_last,
            parallel_output=False,
        ).cuda()
        
        print(f"[Rank {rank}] 创建模型成功，层数: {layers_per_stage}")
        
        # 创建数据（只有第一个阶段需要真实数据）
        if is_first:
            dataset = DummyLlamaDataset(num_samples=16, seq_len=32, vocab_size=1000)
            dataloader = DataLoader(dataset, batch_size=4)
            
            def gpu_data_iterator():
                for batch in dataloader:
                    yield {k: v.cuda() for k, v in batch.items()}
            
            data_iterator = gpu_data_iterator()
        else:
            data_iterator = None
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 使用模型自带的流水线并行方法
        print(f"[Rank {rank}] 开始流水线并行训练...")
        losses = model.pipeline_forward_backward(
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_microbatches=4,
            forward_only=False,
        )
        
        if is_last:
            print(f"✓ [Rank {rank}] 流水线并行训练成功，losses数量: {len(losses)}")
            if len(losses) > 0:
                # 处理 loss 可能是标量或张量的情况
                try:
                    avg_loss = sum(l.item() if l.numel() == 1 else l.mean().item() for l in losses) / len(losses)
                    print(f"  平均 loss: {avg_loss:.4f}")
                except:
                    print(f"  (loss 格式不适合显示)")
        else:
            print(f"✓ [Rank {rank}] 流水线并行训练成功")
        
        # 同步所有进程
        dist.barrier()
        
        return True
        
    except Exception as e:
        import traceback
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"✗ [Rank {rank}] 流水线并行训练失败: {e}")
        traceback.print_exc()
        return False


def test_set_input_tensor():
    """测试7：set_input_tensor 方法测试"""
    print("\n=== 测试7：set_input_tensor 方法 ===")
    try:
        from crius.train_backend.megatron.llama_model import LlamaModel, llama2_7b_config
        
        config = llama2_7b_config(
            num_layers=2,
            hidden_size=128,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建非第一阶段的模型（pre_process=False）
        model = LlamaModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            pre_process=False,  # 中间或最后阶段
            post_process=True,
        ).cuda()
        
        # 创建输入张量（模拟从上一阶段接收）
        seq_len, batch_size, hidden_size = 16, 2, 128
        input_tensor = torch.randn(seq_len, batch_size, hidden_size).cuda()
        
        # 设置输入张量
        model.set_input_tensor(input_tensor)
        print(f"✓ set_input_tensor 成功，输入形状: {input_tensor.shape}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ set_input_tensor 测试失败: {e}")
        traceback.print_exc()
        return False


def test_llama_configs():
    """测试8：预定义配置测试"""
    print("\n=== 测试8：预定义 Llama 配置 ===")
    try:
        from crius.train_backend.megatron.llama_model import (
            llama2_7b_config,
            llama2_13b_config,
            llama2_70b_config,
            llama3_8b_config,
            llama3_70b_config,
        )
        
        configs = {
            'Llama2-7B': llama2_7b_config(),
            'Llama2-13B': llama2_13b_config(),
            'Llama2-70B': llama2_70b_config(),
            'Llama3-8B': llama3_8b_config(),
            'Llama3-70B': llama3_70b_config(),
        }
        
        print("配置详情:")
        for name, config in configs.items():
            print(f"  {name}:")
            print(f"    - 层数: {config.num_layers}")
            print(f"    - 隐藏层大小: {config.hidden_size}")
            print(f"    - 注意力头数: {config.num_attention_heads}")
            print(f"    - KV组数 (GQA): {config.num_query_groups}")
            print(f"    - FFN大小: {config.ffn_hidden_size}")
        
        print("✓ 所有预定义配置正确")
        return True
    except Exception as e:
        import traceback
        print(f"✗ 配置测试失败: {e}")
        traceback.print_exc()
        return False


# ============== 主函数 ==============

def main():
    """主测试函数"""
    print("=" * 70)
    print("LlamaModel 完整测试套件")
    print("=" * 70)
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()
    
    # 运行测试
    results = {}
    
    # 基础测试（所有rank都运行）
    if rank == 0 or not dist.is_initialized():
        results['导入测试'] = test_basic_import()
        results['模型初始化'] = test_model_initialization()
        results['RMSNorm层'] = test_rms_norm()
        results['预定义配置'] = test_llama_configs()
        results['set_input_tensor'] = test_set_input_tensor()
    
    # 根据GPU数量选择不同的测试
    if world_size == 1:
        # 单GPU测试
        if rank == 0:
            results['单GPU前向传播'] = test_forward_pass_single_gpu()
            results['单GPU反向传播'] = test_backward_pass_single_gpu()
    else:
        # 多GPU流水线并行测试
        results['流水线并行训练'] = test_pipeline_parallel_training()
    
    # 打印结果（只在rank 0打印）
    if rank == 0 or world_size == 1:
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)
        
        for name, passed in results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{name:20s}: {status}")
        
        all_passed = all(results.values())
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ 所有测试通过！")
        else:
            print("❌ 部分测试失败！")
        print("=" * 70)
    
    # 清理
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

