import torch
import torch.distributed as dist
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(current_dir)
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from crius.offline.offline_recorder import OfflineDataRecorder
import gc

class CommunicationProfiler:
    def __init__(self, world_size, rank, backend="nccl"):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.setup()

    def setup(self):
        dist.init_process_group(backend=self.backend, world_size=self.world_size, rank=self.rank)
        torch.cuda.set_device(self.rank)

    def cleanup(self):
        dist.destroy_process_group()

    def sync_barrier(self):
        dist.barrier()

    def measure_latency(self, fn, *args, **kwargs):
        self.sync_barrier()
        start = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()  
        end = time.time()
        local_latency_ms = torch.tensor([(end - start) * 1000], dtype=torch.float64, device="cuda")

        dist.all_reduce(local_latency_ms, op=dist.ReduceOp.MAX)
        return local_latency_ms.item()

    def profile_all_reduce(self, tensor_size):
        tensor = torch.ones(tensor_size, device="cuda")
        return self.measure_latency(dist.all_reduce, tensor, dist.ReduceOp.SUM)

    def profile_all_gather(self, tensor_size):
        tensor = torch.ones(tensor_size, device="cuda")
        gather_list = [torch.zeros(tensor_size, device="cuda") for _ in range(self.world_size)]
        return self.measure_latency(dist.all_gather, gather_list, tensor)

    def run_profile(self, tp_degree, tensor_sizes):
        recorder = OfflineDataRecorder()

        for tensor_size in tensor_sizes: 
            all_reduce_latency_ms = self.profile_all_reduce(tensor_size)

            if self.rank == 0:
                query_key = OfflineDataRecorder.get_query_key("All_Reduce", tp_degree, tensor_size)
                recorder._check_cache(query_key, "All_Reduce", tp_degree, tensor_size)

            # Reset
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if self.rank == 0:
                recorder._write_one_entry(query_key, all_reduce_latency_ms, None)

        # for tensor_size in tensor_sizes:
        #     all_gather_latency_ms = self.profile_all_gather(tensor_size)

        #     if self.rank == 0:
        #         query_key = OfflineDataRecorder.get_query_key("All_Gather", tp_degree, tensor_size)
        #         recorder._check_cache(query_key, "All_Gather", tp_degree, tensor_size)

        #     # Reset
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()

        #     if self.rank == 0:
        #         recorder._write_one_entry(query_key, all_gather_latency_ms, None)

        if self.rank == 0:
            recorder._store()

        # return {
        #     "all_reduce_latency_ms": all_reduce_latency * 1000,
        #     "all_gather_latency_ms": all_gather_latency * 1000
        # }



if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    profiler = CommunicationProfiler(world_size, rank)

    tensor_sizes = [
                1024, 4096, 16384, 65536, 262144,  # QKV

                1048576, 4194304, 16777216         # MLP

                # 67108864, 134217728, 268435456     # grad
            ]

    profiler.run_profile(world_size, tensor_sizes)

    # for tensor_size in tensor_sizes:
    #     results = profiler.run_profile(tensor_sizes)

    #     if rank == 0:
    #         # print(f"Results for tensor size {tensor_size}: {results}")
    #         pass

    profiler.cleanup()