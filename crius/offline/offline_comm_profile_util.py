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

__all__ = [ "CommunicationOfflineProfiler", ]


class CommunicationOfflineProfiler:
    """
    CommunicationOfflineProfiler class that measures communication operators latency.

    Args:
        node_num: str: The number of machine.
        world_size: The number of thread on a node.
        rank: The rank of the thread.
        backend: The collective communications library.
    """

    def __init__(self, node_num, world_size, rank, backend="nccl"):
        """ Initialize a CommunicationOfflineProfiler object. """

        self.node_num = node_num
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.setup()

    def setup(self):
        """ Initialize distribution. """
        dist.init_process_group(backend=self.backend, world_size=self.world_size, rank=self.rank)
        torch.cuda.set_device(self.rank)

    def cleanup(self):
        """ Postprocess distribution. """
        dist.destroy_process_group()

    def sync_barrier(self):
        """ Synchronization. """
        dist.barrier()

    def measure_latency(self, fn, *args, **kwargs):
        """ Measure the latency of the communicaton operation, use the maximum. """
        self.sync_barrier()
        start = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()  
        end = time.time()
        local_latency_ms = torch.tensor([(end - start) * 1000], dtype=torch.float16, device="cuda")

        dist.all_reduce(local_latency_ms, op=dist.ReduceOp.MAX)
        return local_latency_ms.item()

    def profile_all_reduce(self, tensor_size, dtype):
        """ Measure the latency of all_gather operation, given tensor_size and dtype . """
        tensor = torch.randn(tensor_size, device="cuda", dtype=dtype)
        return self.measure_latency(dist.all_reduce, tensor, dist.ReduceOp.SUM)

    # def profile_all_gather(self, tensor_size, dtype):
    #     tensor = torch.randn(tensor_size, device="cuda")
    #     gather_list = [torch.zeros(tensor_size, device="cuda") for _ in range(self.world_size)]
    #     return self.measure_latency(dist.all_gather, gather_list, tensor)

    def run_profile(self, node_num, group_size, tensor_sizes, dtype):
        """ Launch the communication offline profiling. """
        recorder = OfflineDataRecorder(
            to_store_path = "./crius/offline/data",
            data_filename = "prof_comm_data.pkl",
            force_overwrite = True
        )

        for tensor_size in tensor_sizes: 
            all_reduce_latency_ms = self.profile_all_reduce(tensor_size, dtype)

            if self.rank == 0:
                query_key = OfflineDataRecorder.get_query_key("all_reduce", None, None, None, None, None, node_num, group_size, tensor_size)
                recorder._check_cache(query_key, "all_reduce", node_num, group_size, tensor_size)

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
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    node_num = (world_size - 1) // local_world_size + 1

    profiler = CommunicationOfflineProfiler(node_num, world_size, rank)

    tensor_sizes = [
                1024, 4096, 8192, 
                
                16384, 24576, 32768, 49152, 65536, 
                
                262144, 640000, 
                
                1280000, 1572864, 2097152, 8388608, 1048576, 4194304, 
                
                16777216, 25165824, 25600000, 33554432,      

                67108864, 134217728, 268435456, 11539607552     
            ]

    profiler.run_profile(node_num, world_size, tensor_sizes, torch.float16)

    # for tensor_size in tensor_sizes:
    #     results = profiler.run_profile(tensor_sizes)

    #     if rank == 0:
    #         # print(f"Results for tensor size {tensor_size}: {results}")
    #         pass

    profiler.cleanup()