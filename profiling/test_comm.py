import os
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity


def all_gather_ag(rank, world_size, warmup, average, torchsize):

    local_tensor = torch.rand(torchsize).to(rank)
    output_tensors = [torch.empty(torchsize).to(rank) for _ in range(world_size)]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # try:
    #     with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         # 前向计算
    #         dist.all_gather(output_tensors, local_tensor)
    #         dist.barrier()
    # except Exception as e:
    #     print(f"An error occurred during forward pass: {e}")
    # else:
    #     # 分析通信算子
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    for i in range(warmup + average):
        if i == warmup - 1:
            start_event.record()
        elif i == warmup + average - 1:
            torch.cuda.synchronize()
            end_event.record()
            All_gather_time = start_event.elapsed_time(end_event) / average
            print(f'The average time of All_gather_time is {All_gather_time} ms\n')
        dist.all_gather(output_tensors, local_tensor)
        dist.barrier()

def all_gather_sr(rank, warmup, average, torchsize):

    local_tensor = torch.rand(torchsize).to(rank)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(warmup + average):
        if i == warmup - 1:
            start_event.record()
        elif i == warmup + average - 1:
            torch.cuda.synchronize()
            end_event.record()
            All_gather_time = start_event.elapsed_time(end_event) / average
            print(f'The average time of All_gather_time is {All_gather_time} ms\n')
        if rank == 0:
            dist.send(tensor=local_tensor, dst=1)
            print(f"rank {rank} tensor have send ")
            dist.recv(tensor=local_tensor, src=1)
            print(f"rank {rank} tensor have recv ")
        elif rank == 1:
            dist.recv(tensor=local_tensor, src=0)
            print(f"rank {rank} tensor have recv ")
            dist.send(tensor=local_tensor, dst=0)
            print(f"rank {rank} tensor have send ")
        dist.barrier()


def reduce_scatter(rank, world_size, warmup, average, torchsize):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    tensor_list = [torch.randn(torchsize).to(rank) for _ in range(world_size)]
    output = torch.zeros(torchsize // world_size).to(rank)

    # 开始计时
    start_event.record()

    for i in range(warmup + average):
        if i == warmup - 1:
            start_event.record()
        elif i == warmup + average - 1:
            torch.cuda.synchronize()
            end_event.record()
            reduce_scatter_time = start_event.elapsed_time(end_event) / average
            print(f'The average time of reduce_scatter_time is {reduce_scatter_time} ms\n')
        # 执行 reduce-scatter 操作
        dist.reduce_scatter(output, tensor_list)
        dist.barrier()

if __name__ == "__main__":

    # 初始化进程组
    dist.init_process_group(backend='nccl')

    # 获取当前进程rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 设置张量和梯度大小
    torchsize = 3552402
    batchsize = 32
    gradient_size = torchsize * world_size

    # 设定测试warmup和average
    warmup = 5
    average = 5
    # all_gather_ag(rank, warmup, average, torchsize)
    # all_gather_sr(rank, warmup, average, torchsize)
    reduce_scatter(rank, warmup, average, torchsize)

    dist.destroy_process_group()

