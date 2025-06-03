#!/usr/bin/env python
"""
test_FT_with_dymano.py

示例：
    python test_FT_with_dymano.py --model /app/profiling/model/qwen-7b-chat --device cuda:0
"""

import argparse
import time
import gc
import contextlib

import torch
import transformers
from torch._subclasses import FakeTensorMode
import torch._dynamo as dynamo
torch._dynamo.config.suppress_errors = True
from typing import List

# 启用未来特性（确保参数转换时的设备一致性）
# 有了这行代码，下述.to(device)真的会把每个参数的device字段改成 cuda:0 (仍然是FakeTensor，不占显存)
torch.__future__.set_swap_module_params_on_conversion(True)

# ───────────────────────── helpers ──────────────────────────
@contextlib.contextmanager
def measure_time(tag: str):
    """简单计时器"""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{tag} took {dt:.2f} s")


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable


@torch.compile(backend=my_compiler)
def forward(model, x):
    return model(x)


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(768, 1536)
        self.linear2 = torch.nn.Linear(1536, 1536)
        self.linear3 = torch.nn.Linear(1536, 768)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out).clamp(min=0.0, max=1.0)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local LLM model path")
    parser.add_argument("--device", default="cuda", help="e.g. cuda:0 / cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = args.device
    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    batch_size, seq_len, dim = 1, 10, 4096
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # 创建全局 FakeTensorMode 上下文
    with FakeTensorMode(allow_non_fake_inputs=True) as global_fake_mode:
        print("\n=== FakeTensor model ===")

        # ── 模型加载 ──
        with measure_time("Load fake model"):
            cfg = transformers.AutoConfig.from_pretrained(
                args.model, local_files_only=True, trust_remote_code=True
            )
            model = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
            model.to(device)


        """
        这里发现在运行compiled = torch.compile(model, backend=my_compiler)时，不把backend设置为inductor就能执行：
        但是打印的信息非常混乱
        """
        # ── Dynamo 编译 ──
        with measure_time("Dynamo compile & run (fake)"):
            # # 定义自定义backend，包含统计信息收集
            # stats_dict = {"graphs_generated": 0}
            
            # def stats_backend(gm, example_inputs):
            #     stats_dict["graphs_generated"] += 1
            #     print(f"Generated graph {stats_dict['graphs_generated']}")
            #     # 使用正确的 backend 导入方式
            #     from torch._dynamo.backends import eager
            #     return eager(gm, example_inputs)

            # 使用自定义backend进行编译
            compiled = torch.compile(model, backend=my_compiler)

            # 预热和推理
            with torch.no_grad():
                _ = compiled(input_ids=input_ids, attention_mask=attention_mask)
                _ = compiled(input_ids=input_ids, attention_mask=attention_mask)
        

        '''
        FX 图打印在实例化 MyModule()这样的简单模型不会报错，但如果是加载大模型（如 LLaMA、Qwen 等），会报错：
        torch._dynamo.exc.TorchRuntimeError: Failed running call_function <built-in function mul>(*(FakeTensor(..., size=(1, 10, 32, 128), grad_fn=<AliasBackward0>), FakeTensor(..., device='cuda:0', size=(1, 10, 1, 128))), **{}):
        Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two different devices cpu, cuda:0
        '''
        # # ── FX 图打印 ──
        # print("\n[FX] graph overview")
        # # Use concrete args to avoid input validation errors
        # concrete_args = {
        #     'inputs_embeds': input_ids,
        #     'attention_mask': attention_mask,
        #     'position_ids': None,
        #     'past_key_values': None,
        #     'token_type_ids': None,
        #     'return_dict': True,
        #     'output_attentions': None,
        #     'output_hidden_states': None,
        # }
        # gm = torch.fx.symbolic_trace(model, concrete_args=concrete_args)
        # gm.graph.print_tabular()
        
        # print("\n[FX] call_module nodes")
        # for n in gm.graph.nodes:
        #     if n.op == "call_module":
        #         sub = gm.get_submodule(n.target)
        #         print(f"{n.target:<50} {sub.__class__.__name__}")
        
        # # ── Dynamo 图打印 ──
        # print("\n[Dynamo] graph export")
        # with torch.no_grad():
        #     gm_dynamo, guards = dynamo.export(model)(input_ids, attention_mask=attention_mask)
        
        # print("\n[Dynamo] guards")
        # for g in guards:
        #     print(" ", g)
        
        # print("\n[Dynamo] graph code ↓\n")
        # print(gm_dynamo.code)
        
        # print("\n[Dynamo] call_module nodes")
        # for n in gm_dynamo.graph.nodes:
        #     if n.op == "call_module":
        #         sub = gm_dynamo.get_submodule(n.target)
        #         print(f"{n.target:<50} {sub.__class__.__name__}")
    
    # 清理资源（上下文退出后自动处理 FakeTensorMode）
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()




# #!/usr/bin/env python
# """
# test_fake_tensor_fx_dynamo.py

# 示例：
#     python test_FT_with_dynamo.py --model D:/LLM/qwen1.5-1.8b --device cuda:0
# """

# import argparse
# import time
# import gc
# import contextlib

# import torch
# import transformers
# from torch._subclasses import FakeTensorMode
# from torch.fx import symbolic_trace
# import torch._dynamo as dynamo

# # ───────────────────────── helpers ──────────────────────────
# @contextlib.contextmanager
# def measure_time(tag: str):
#     """简单计时器"""
#     t0 = time.perf_counter()
#     yield
#     dt = time.perf_counter() - t0
#     print(f"{tag} took {dt:.2f} s")

# def load_normal(path: str, device: str, dtype):
#     """常规真张量加载"""
#     cfg = transformers.AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
#     model = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
#     model.to(device)
#     return model

# torch.__future__.set_swap_module_params_on_conversion(True)
# def load_fake(path: str, device: str, dtype):
#     """FakeTensorMode 下加载（零显存）"""
#     with FakeTensorMode():
#         cfg = transformers.AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
#         model = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
#         model.to(device)
#     return model


# def test_dynamo_compilation(model, device, batch_size: int = 1, seq_len: int = 10):
#     """演示 torch.compile 后的推理耗时与统计"""
#     input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
#     attention_mask = torch.ones(batch_size, seq_len, device=device)

#     compiled = torch.compile(model)

#     with torch.no_grad():
#         _ = compiled(input_ids=input_ids, attention_mask=attention_mask)   # 预热

#     with measure_time("Compiled inference"):
#         with torch.no_grad():
#             _ = compiled(input_ids=input_ids, attention_mask=attention_mask)

#     if hasattr(compiled, "_torchdynamo"):
#         stats = compiled._torchdynamo.stats
#         print("Dynamo stats:")
#         for k, v in stats.items():
#             print(f"  {k:<25}: {v}")
#     return compiled

# # ─────────────────────── FX / Dynamo 打印 ────────────────────────
# def print_layers_via_fx(model):
#     """symbolic_trace 后列出 call_module 节点"""
#     with FakeTensorMode():
#         gm = symbolic_trace(model)
#     print("\n[FX] graph overview")
#     gm.graph.print_tabular()
#     print("\n[FX] call_module nodes")
#     for n in gm.graph.nodes:
#         if n.op == "call_module":
#             sub = gm.get_submodule(n.target)
#             print(f"{n.target:<50} {sub.__class__.__name__}")

# def print_layers_via_dynamo(model, device, batch_size: int = 1, seq_len: int = 10):
#     """torch._dynamo.export 后列出 call_module 节点与代码"""
#     input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
#     attention_mask = torch.ones(batch_size, seq_len, device=device)

#     with FakeTensorMode():
#         gm, guards = dynamo.export(model, input_ids, attention_mask)

#     print("\n[Dynamo] guards")
#     for g in guards:
#         print(" ", g)

#     print("\n[Dynamo] graph code ↓\n")
#     print(gm.code)

#     print("\n[Dynamo] call_module nodes")
#     for n in gm.graph.nodes:
#         if n.op == "call_module":
#             sub = gm.get_submodule(n.target)
#             print(f"{n.target:<50} {sub.__class__.__name__}")

# # ───────────────────────────── main ──────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True, help="Local HF model path")
#     parser.add_argument("--device", default="cuda", help="e.g. cuda:0 / cpu")
#     args = parser.parse_args()

#     torch.manual_seed(0)
#     device = args.device
#     dtype = (
#         torch.bfloat16
#         if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
#         else torch.float16
#     )

#     # # —— 真张量模型 ——
#     # print("\n=== Normal (real tensor) model ===")
#     # with measure_time("Load real model"):
#     #     normal_model = load_normal(args.model, device, dtype)
#     # with measure_time("Dynamo compile & run (real)"):
#     #     test_dynamo_compilation(normal_model, device)
#     # print_layers_via_fx(normal_model)
#     # print_layers_via_dynamo(normal_model, device)
#     # del normal_model; gc.collect(); torch.cuda.empty_cache()

#     # —— 假张量模型 ——
#     print("\n=== FakeTensor model ===")
#     with measure_time("Load fake model"):
#         fake_model = load_fake(args.model, device, dtype)
#     with measure_time("Dynamo compile & run (fake)"):
#         test_dynamo_compilation(fake_model, device)
#     print_layers_via_fx(fake_model)
#     print_layers_via_dynamo(fake_model, device)
#     del fake_model; gc.collect(); torch.cuda.empty_cache()

# if __name__ == "__main__":
#     # 示例：
#     #     python test_FT_with_dynamo.py --model D:/LLM/qwen1.5-1.8b --device cuda:0
#     main()


