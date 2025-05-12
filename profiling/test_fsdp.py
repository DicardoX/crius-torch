import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from test_comm import reduce_scatter, all_gather_sr, all_gather_ag


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Some configuration parameters"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="The index of rounds that need to be omitted.")

    parser.add_argument(
        "--average",
        type=int,
        default=5,
        help="The number of rounds that need to be iterated after warmup.")

    args = parser.parse_args()
    return args

# 1. 创建虚拟数据集
class SimpleDataset(Dataset):
    def __init__(self, num_samples, seq_len, dim):
        self.data = torch.randn(num_samples, seq_len, dim)
        self.labels = torch.randint(0, 2, (num_samples,)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple module for demonstration
class MyModel(torch.nn.Module):
    def __init__(self, seq_len, dim) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, 2*dim)
        self.linear2 = torch.nn.Linear(2*dim, 2*dim)
        self.linear3 = torch.nn.Linear(2*dim, 2*dim)
        self.linear4 = torch.nn.Linear(2*dim, dim)
        self.linear5 = torch.nn.Linear(dim, 2)
        self.linear6 = torch.nn.Linear(seq_len*2, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        out =out.view(out.size(0), -1)
        logits = self.linear6(out)
        return logits

def build_model():
    model = MyModel()
    return model

def FSDP_Train(rank, num_samples, seq_len, dim, batch, world_size, warmup, average):
    # 创建数据集和数据加载器
    print(f"rank{rank} fsdp Dataset build")
    dataset = SimpleDataset(num_samples, seq_len, dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch, sampler=sampler)
    print(f"rank{rank} DataLoader build")

    # 创建模型并使用 FSDP 封装
    model = MyModel(seq_len, dim).to(rank)
    fsdp_model = FSDP(model, device_id=rank)
    print(f"rank{rank} FSDP model build")
    # # 模型划分的分析
    num_param = None
    for name, param in fsdp_model.named_parameters():
        num_param = param.numel()
        print(f"  Parameter: {name}, fsdp unit_size: {param.shape}, Device: {param.device}")

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 设置CUDA.Event()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"rank{rank} FSDP training start")
    # 训练循环,测试FSDP平均前向时间
    for i in range(warmup + average):
        if i == warmup - 1:
            start_event.record()
        elif i == warmup + average - 1:
            torch.cuda.synchronize()
            end_event.record()
            single_forward_time = start_event.elapsed_time(end_event) / average
            print(f'The average time of rank{rank} single_forward_time is {single_forward_time}ms')

        # 设置数据
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(rank)
        labels = labels.to(rank)

        # 前向传播
        outputs = fsdp_model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("training end")
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(rank)
    labels = labels.to(rank)
    print(f"rank{rank} FSDP batch size==  {inputs.shape[0]}")


    try:
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            # 前向计算
            outputs = fsdp_model(inputs).squeeze()
            loss = criterion(outputs, labels)
            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(f"An error occurred during forward pass: {e}")
    else:
        # 分析通信算子
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def Single_Train(num_samples, seq_len, dim, batch, warmup, average):
    # # 构建数据集
    dataset = SimpleDataset(num_samples, seq_len, dim)
    dataloader = DataLoader(dataset, batch_size=batch)
    print(f"single DataLoader build")

    # 配置GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"发现 {num_gpus} 个可用的 GPU.")
        # 可以选择特定的 GPU 设备，这里选择第一个 GPU（索引从 0 开始）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("没有可用的 GPU，使用 CPU 进行计算.")

    # single device model build
    print(f"single model build")
    model = MyModel(seq_len, dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 设置CUDA.Event()
    single_forward_time = None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 训练循环,测试单卡
    print(f"single training start")
    for i in range(warmup + average):
        if i == warmup - 1:
            start_event.record()
        elif i == warmup + average - 1:
            torch.cuda.synchronize()
            end_event.record()
            single_forward_time = start_event.elapsed_time(end_event) / average
            print(f'The average time of rank single_forward_time is {single_forward_time}ms')

        # 设置数据
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(f"single batch size ==  {inputs.shape[0]}")

        # 前向传播
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    # 初始化进程组, torchrun --nproc-per-node 2 test_fsdp.py
    # dist.init_process_group(backend='nccl')
    # # 设定测试warmup和average
    # warmup = 5
    # average = 5
    # # 测量FSDP单次迭代的训练时长
    # num_samples = 1000
    # seq_len = 10000
    # dim = 768
    # batch = 32
    # num_epochs = warmup + average
    # # 获取当前进程rank
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # torch.cuda.set_device(rank)
    # FSDP_Train(rank, num_samples, seq_len, dim, batch, world_size, warmup, average)
    # dist.destroy_process_group()


    # 测量FSDP单次迭代的训练时长
    # warmup = 5
    # average = 5
    # num_samples = 1000
    # seq_len = 10000
    # dim = 768
    # batch = 32
    # num_epochs = warmup + average
    # Single_Train(num_samples, seq_len, dim, batch, warmup, average)



    # # 通信算子reduce_scatter的测量, unit_size需根据模型参数量的实际大小具体来确定，容易获取
    dist.init_process_group(backend='nccl')
    unit_size = 3552402
    warmup = 5
    average = 5
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    reduce_scatter(rank, world_size, warmup, average, unit_size)
    dist.destroy_process_group()

    # 通信算子all-gather的测量
    # dist.init_process_group(backend='nccl')
    # unit_size = 3552402
    # warmup = 5
    # average = 5
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # torch.cuda.set_device(rank)
    # all_gather_ag(rank, world_size, warmup, average, unit_size)
    # dist.barrier()
    # dist.destroy_process_group()







