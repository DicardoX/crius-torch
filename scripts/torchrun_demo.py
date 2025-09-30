"""
Torchrun AllReduce Demo

运行命令：
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 torchrun_demo.py
"""

import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    print(f"进程启动 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] 使用GPU {local_rank}")
    
    tensor = torch.ones(4, device=device) * (rank + 1)
    print(f"[Rank {rank}] 初始张量: {tensor.tolist()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] AllReduce求和后: {tensor.tolist()}")
    
    dist.barrier()
    
    if rank == 0:
        print("AllReduce演示完成！")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()