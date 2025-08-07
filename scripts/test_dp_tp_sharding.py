import os
import argparse
import torch
from torch import nn
import torch.distributed as dist


bs = 16
seq_len = 2048
hidden_size = 4096
num_shards = 4


def gemm_profile(gemm_op: nn.Module, inputs: torch.Tensor) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(100):
        _ = gemm_op(inputs)    
    torch.cuda.synchronize()
    # Measure
    start_event.record()
    for _ in range(1000):
        _ = gemm_op(inputs)
    end_event.record()
    torch.cuda.synchronize()
    # Return
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / 1000
    return avg_latency_ms


def allreduce_profile(data: torch.Tensor, rank: int) -> float:
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    data = data.to(device) 

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(100):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    # Measure
    start_event.record()
    for _ in range(1000):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    end_event.record()    

    torch.cuda.synchronize(device)
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / 1000

    if dist.get_rank() == 0:
        print(f"All-reduce latency: {avg_latency_ms:.2f} ms")
    
    dist.destroy_process_group()    
    return avg_latency_ms


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    profile_gemm = True
    if profile_gemm:
        torch.cuda.set_device(0)
        # ---------->  DP: slicing along batch dimension
        gemm_op = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float16, device="cuda")
        inputs_dp = torch.randn(bs // num_shards, seq_len, hidden_size, dtype=torch.float16, device="cuda")
        latency_dp = gemm_profile(gemm_op, inputs_dp)
        print(f"DP: {latency_dp:.2f} ms")

        # ---------->  TP: slicing along hidden dimension
        gemm_op = nn.Linear(hidden_size, hidden_size // num_shards, bias=False, dtype=torch.float16, device="cuda")
        inputs_tp = torch.randn(bs, seq_len, hidden_size, dtype=torch.float16, device="cuda")
        latency_tp = gemm_profile(gemm_op, inputs_tp)
        print(f"TP: {latency_tp:.2f} ms")

    else:
        allreduce_data = torch.randn(bs, seq_len, hidden_size // num_shards, dtype=torch.float16)
        allreduce_profile(allreduce_data, local_rank)
