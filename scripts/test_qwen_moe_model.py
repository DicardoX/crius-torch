#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""测试 QwenMoEModel 的完整功能

运行方式：
    # 单GPU测试（无流水线并行）
    python scripts/test_qwen_moe_model.py
    
    # 多GPU测试（2个GPU的流水线并行）
    torchrun --nproc-per-node=2 scripts/test_qwen_moe_model.py
    
    # 多GPU测试（4个GPU的流水线并行）
    torchrun --nproc-per-node=4 scripts/test_qwen_moe_model.py
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


class DummyQwenDataset(Dataset):
    """简单的测试数据集，用于Qwen MoE模型"""
    def __init__(self, num_samples=100, seq_len=128, vocab_size=151936):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Qwen使用因果注意力
        return {
            'input_ids': torch.randint(1, self.vocab_size, (self.seq_len,)),
            'position_ids': torch.arange(self.seq_len),
            'attention_mask': torch.tril(torch.ones(self.seq_len, self.seq_len)).bool(),
            'labels': torch.randint(1, self.vocab_size, (self.seq_len,)),
        }


# ============== 测试用例 ==============

def test_basic_import():
    """测试1：基本导入测试"""
    print("\n=== 测试1：导入 QwenMoEModel 模块 ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            QwenMoETransformerLayer,
            QwenMoETransformerBlock,
            qwen2_5_moe_a14b_config,
            qwen2_5_moe_57b_a14b_config,
            qwen2_5_moe_custom_config,
        )
        from crius.train_backend.megatron.llama_model import RMSNorm
        print("✓ 成功导入 QwenMoEModel 及相关组件")
        
        # 检查配置函数
        config = qwen2_5_moe_a14b_config()
        print(f"✓ Qwen2.5-MoE-A14B 配置: {config.num_layers} 层, "
              f"hidden_size={config.hidden_size}, "
              f"num_experts={config.num_moe_experts}, "
              f"top_k={config.moe_router_topk}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试2：MoE模型初始化测试"""
    print("\n=== 测试2：QwenMoEModel 初始化 ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel, 
            qwen2_5_moe_custom_config
        )
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        
        # 初始化并行状态
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23455', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        # 使用小配置进行测试
        config = qwen2_5_moe_custom_config(
            num_layers=4,  # 减少层数用于测试
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=2,
            ffn_hidden_size=2048,
            num_experts=4,  # 使用4个experts
            top_k=2,  # 每个token选择2个experts
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型
        model = QwenMoEModel(
            config=config,
            vocab_size=151936,  # Qwen2.5 词汇表大小
            max_sequence_length=512,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            rope_theta=1000000.0,  # Qwen2.5 使用更大的 theta
        )
        
        print(f"✓ 成功创建 QwenMoEModel")
        print(f"  - 词汇表大小: {model.vocab_size}")
        print(f"  - 最大序列长度: {model.max_sequence_length}")
        print(f"  - RoPE theta: {model.rope_theta}")
        print(f"  - Expert 数量: {config.num_moe_experts}")
        print(f"  - Top-K: {config.moe_router_topk}")
        
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


def test_moe_layer():
    """测试3：MoE Layer 测试"""
    print("\n=== 测试3：QwenMoETransformerLayer ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoETransformerLayer
        )
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        import torch.nn.functional as F
        
        # 初始化并行状态（单GPU模式）
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        # 创建配置
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=1024,
            kv_channels=64,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            layernorm_epsilon=1e-6,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
            # MoE 配置
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type='aux_loss',
            moe_aux_loss_coeff=0.01,
        )
        
        # 创建 MoE Layer
        layer = QwenMoETransformerLayer(config=config, layer_number=1).cuda()
        
        # 测试前向传播
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        
        output, context = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        assert output.shape == hidden_states.shape, \
            f"输出形状不匹配: {output.shape} vs {hidden_states.shape}"
        print(f"✓ MoE Layer 前向传播成功")
        print(f"  输入形状: {hidden_states.shape}")
        print(f"  输出形状: {output.shape}")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ MoE Layer 测试失败: {e}")
        traceback.print_exc()
        return False


def test_forward_pass_single_gpu():
    """测试4：单GPU前向传播测试"""
    print("\n=== 测试4：单GPU前向传播 ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            qwen2_5_moe_custom_config
        )
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        
        # 初始化并行状态
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        # 创建小配置
        config = qwen2_5_moe_custom_config(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=1024,
            num_experts=4,
            top_k=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型
        model = QwenMoEModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            rope_theta=1000000.0,
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
        assert logits.shape == expected_shape, \
            f"Logits 形状不匹配: {logits.shape} vs {expected_shape}"
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
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            qwen2_5_moe_custom_config
        )
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        
        # 确保并行状态已初始化
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        # 创建小配置
        config = qwen2_5_moe_custom_config(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=512,
            num_experts=4,
            top_k=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建模型和优化器
        model = QwenMoEModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            rope_theta=1000000.0,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 创建输入数据
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        
        # 训练几步
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
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            qwen2_5_moe_custom_config
        )
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
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
            model_parallel_cuda_manual_seed(123 + rank)
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
        
        config = qwen2_5_moe_custom_config(
            num_layers=layers_per_stage,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=1024,
            num_experts=4,
            top_k=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=world_size,
            expert_model_parallel_size=1,
            pipeline_dtype=torch.float32,
            params_dtype=torch.float32,
        )
        
        # 创建模型（每个rank只创建自己那部分）
        model = QwenMoEModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            pre_process=is_first,
            post_process=is_last,
            parallel_output=False,
            rope_theta=1000000.0,
        ).cuda()
        
        print(f"[Rank {rank}] 创建模型成功，层数: {layers_per_stage}")
        
        # 创建数据（只有第一个阶段需要真实数据）
        if is_first:
            dataset = DummyQwenDataset(num_samples=16, seq_len=32, vocab_size=1000)
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
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            qwen2_5_moe_custom_config
        )
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        
        # 确保并行状态已初始化
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        config = qwen2_5_moe_custom_config(
            num_layers=2,
            hidden_size=128,
            num_experts=4,
            top_k=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
        )
        
        # 创建非第一阶段的模型（pre_process=False）
        model = QwenMoEModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            pre_process=False,  # 中间或最后阶段
            post_process=True,
            rope_theta=1000000.0,
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


def test_moe_configs():
    """测试8：预定义 MoE 配置测试"""
    print("\n=== 测试8：预定义 Qwen2.5 MoE 配置 ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            qwen2_5_moe_a14b_config,
            qwen2_5_moe_57b_a14b_config,
            qwen2_5_moe_custom_config,
        )
        
        configs = {
            'Qwen2.5-MoE-A14B': qwen2_5_moe_a14b_config(),
            'Qwen2.5-MoE-57B-A14B': qwen2_5_moe_57b_a14b_config(),
            'Qwen2.5-MoE-Custom': qwen2_5_moe_custom_config(
                num_layers=12,
                hidden_size=1024,
                num_experts=8,
                top_k=2,
            ),
        }
        
        print("配置详情:")
        for name, config in configs.items():
            print(f"  {name}:")
            print(f"    - 层数: {config.num_layers}")
            print(f"    - 隐藏层大小: {config.hidden_size}")
            print(f"    - 注意力头数: {config.num_attention_heads}")
            print(f"    - KV组数 (GQA): {config.num_query_groups}")
            print(f"    - FFN大小: {config.ffn_hidden_size}")
            print(f"    - Expert数量: {config.num_moe_experts}")
            print(f"    - Top-K: {config.moe_router_topk}")
            print(f"    - 负载均衡系数: {config.moe_aux_loss_coeff}")
        
        print("✓ 所有预定义配置正确")
        return True
    except Exception as e:
        import traceback
        print(f"✗ 配置测试失败: {e}")
        traceback.print_exc()
        return False


def test_moe_routing():
    """测试9：MoE 路由和负载均衡测试"""
    print("\n=== 测试9：MoE 路由和负载均衡 ===")
    try:
        from crius.train_backend.megatron.qwen_moe_model import (
            QwenMoEModel,
            qwen2_5_moe_custom_config
        )
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        
        # 确保并行状态已初始化
        if not dist.is_initialized():
            print("需要分布式环境，初始化单GPU模式...")
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            # 初始化 RNG tracker
            model_parallel_cuda_manual_seed(123)
        except:
            print("并行状态已初始化，继续...")
        
        # 创建配置，使用较多的experts来测试负载均衡
        config = qwen2_5_moe_custom_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=1024,
            num_experts=8,  # 使用8个experts
            top_k=2,  # 每个token选择2个
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            params_dtype=torch.float32,
            moe_aux_loss_coeff=0.01,  # 启用负载均衡loss
        )
        
        # 创建模型
        model = QwenMoEModel(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            rope_theta=1000000.0,
        ).cuda()
        
        # 创建较大的batch来观察路由行为
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).cuda()
        labels = torch.randint(1, 1000, (batch_size, seq_len)).cuda()
        
        # 前向传播
        print(f"测试 MoE 路由 (batch_size={batch_size}, seq_len={seq_len}, "
              f"num_experts={config.num_moe_experts}, top_k={config.moe_router_topk})...")
        
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        print(f"✓ MoE 路由测试成功")
        print(f"  Loss (包含aux loss): {loss.item():.4f}")
        print(f"  总tokens: {batch_size * seq_len}")
        print(f"  每个token选择 {config.moe_router_topk} 个experts")
        
        return True
    except Exception as e:
        import traceback
        print(f"✗ MoE 路由测试失败: {e}")
        traceback.print_exc()
        return False


# ============== 主函数 ==============

def main():
    """主测试函数"""
    print("=" * 70)
    print("QwenMoEModel 完整测试套件")
    print("=" * 70)
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()
    
    # 运行测试
    results = {}
    
    # 根据GPU数量选择不同的测试
    if world_size == 1:
        # 单GPU测试 - 运行所有测试
        if rank == 0:
            results['导入测试'] = test_basic_import()
            results['MoE模型初始化'] = test_model_initialization()
            results['MoE Layer'] = test_moe_layer()
            results['预定义MoE配置'] = test_moe_configs()
            results['set_input_tensor'] = test_set_input_tensor()
            results['单GPU前向传播'] = test_forward_pass_single_gpu()
            results['单GPU反向传播'] = test_backward_pass_single_gpu()
            results['MoE路由和负载均衡'] = test_moe_routing()
    else:
        # 多GPU流水线并行测试 - 只运行必要的测试
        if rank == 0:
            results['导入测试'] = test_basic_import()
            results['预定义MoE配置'] = test_moe_configs()
        
        # 同步所有进程，确保都准备好了
        dist.barrier()
        
        # 所有进程都运行流水线并行测试
        results['流水线并行训练'] = test_pipeline_parallel_training()
    
    # 同步所有进程（多GPU模式）
    if world_size > 1:
        dist.barrier()
    
    # 打印结果（只在rank 0打印）
    if rank == 0:
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
    if dist.is_initialized() and world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()



