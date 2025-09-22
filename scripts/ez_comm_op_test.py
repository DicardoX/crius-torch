import torch
import torch.distributed as dist
import time
import argparse
import numpy as np
from typing import List, Tuple

import os
os.environ['NCCL_SHM_DISABLE'] = '1'  
os.environ['NCCL_P2P_DISABLE'] = '1'  
os.environ['NCCL_IB_DISABLE'] = '1'   

def setup(rank, world_size):
    """init-dist"""
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """clear-dist"""
    dist.destroy_process_group()

def benchmark_all_reduce(rank, world_size, tensor_sizes: List[int], num_iterations: int = 100):
    """base all_reduce"""
    results = {}
    
    for size in tensor_sizes:
        # create tensor
        tensor = torch.randn(size, device=f"cuda:{rank}")
        
        # warmup
        for _ in range(10):
            dist.all_reduce(tensor)
        
        # sync
        dist.barrier()
        
        # timing
        start_time = time.time()
        for _ in range(num_iterations):
            dist.all_reduce(tensor)
        dist.barrier()
        end_time = time.time()
        
        # cal average latency(ms)
        avg_latency = (end_time - start_time) * 1000 / num_iterations
        results[size] = avg_latency
        
        if rank == 0:
            print(f"AllReduce - Size: {size}, Avg Latency: {avg_latency:.4f} ms")
    
    return results

def benchmark_all_gather(rank, world_size, tensor_sizes: List[int], num_iterations: int = 100):
    """base all_gather"""
    results = {}
    
    for size in tensor_sizes:
        # create tensor
        tensor = torch.randn(size, device=f"cuda:{rank}")
        output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        # warmup
        for _ in range(10):
            dist.all_gather(output_tensors, tensor)
        
        # sync
        dist.barrier()
        
        # timing
        start_time = time.time()
        for _ in range(num_iterations):
            dist.all_gather(output_tensors, tensor)
        dist.barrier()
        end_time = time.time()
        
        # cal average latency(ms)
        avg_latency = (end_time - start_time) * 1000 / num_iterations
        results[size] = avg_latency
        
        if rank == 0:
            print(f"AllGather - Size: {size}, Avg Latency: {avg_latency:.4f} ms")
    
    return results

def run_benchmark(rank, world_size, args):
    """run base test"""
    setup(rank, world_size)
    
    # tensor size
    tensor_sizes = [
        1024,        # 1K
        10240,       # 10K
        102400,      # 100K
        1024000,     # 1M
        10240000,    # 10M
    ]
    
    if rank == 0:
        print(f"=== Benchmark with World Size: {world_size} ===")
        print(f"Number of iterations: {args.iterations}")
        print(f"Tensor sizes: {tensor_sizes}")
        print("-" * 50)
    
    # all_reduce test
    if rank == 0:
        print("\nTesting all_reduce...")
    all_reduce_results = benchmark_all_reduce(rank, world_size, tensor_sizes, args.iterations)
    
    # all_gather test
    if rank == 0:
        print("\nTesting all_gather...")
    all_gather_results = benchmark_all_gather(rank, world_size, tensor_sizes, args.iterations)
    
    # gather at rank0
    if rank == 0:
        print("\n" + "="*50)
        print("SUMMARY RESULTS:")
        print("="*50)
        print("\nall_reduce Results:")
        for size, latency in all_reduce_results.items():
            print(f"Size: {size:8d} | Latency: {latency:.4f} ms")
        
        print("\nall_gather Results:")
        for size, latency in all_gather_results.items():
            print(f"Size: {size:8d} | Latency: {latency:.4f} ms")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.distributed collective operations")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    args = parser.parse_args()
    
    # multi-proc
    torch.multiprocessing.spawn(
        run_benchmark,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()