import torch
import torch.distributed as dist
import time
import os
import sys

import gc
from crius.offline import OfflineProfileConfigs, CriusOfflineDataManager

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
        return self.measure_latency(OfflineProfileConfigs.allreduce_op_func, tensor, dist.ReduceOp.SUM)

    # def profile_all_gather(self, tensor_size, dtype):
    #     tensor = torch.randn(tensor_size, device="cuda")
    #     gather_list = [torch.zeros(tensor_size, device="cuda") for _ in range(self.world_size)]
    #     return self.measure_latency(dist.all_gather, gather_list, tensor)

    def run_profile(self, nodes_num, group_size):
        """ Launch the communication offline profiling. """
        offline_data_manager = CriusOfflineDataManager(
            profile_configs= OfflineProfileConfigs,
            to_store_path = "./crius/offline/data/",
            data_filename = "prof_data.pkl",
        )

        # tensor_sizes = OfflineProfileConfigs.comm_data_sizes
        tensor_sizes = [2**i for i in range(1,31)]
        dtype = OfflineProfileConfigs.dtype

        for tensor_size in tensor_sizes: 
            all_reduce_latency_ms = self.profile_all_reduce(tensor_size, dtype)

            if self.rank == 0:
                query_key = CriusOfflineDataManager.get_query_key("all_reduce", None, None, None, None, None, nodes_num, group_size, tensor_size)
                # offline_data_manager._check_cache(query_key, "all_reduce", nodes_num, group_size, tensor_size)

            # Reset
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if self.rank == 0:
                offline_data_manager._write_one_entry(query_key, all_reduce_latency_ms, None)

        # for tensor_size in tensor_sizes:
        #     all_gather_latency_ms = self.profile_all_gather(tensor_size)

        #     if self.rank == 0:
        #         query_key = CriusOfflineDataManager.get_query_key("All_Gather", tp_degree, tensor_size)
        #         offline_data_manager._check_cache(query_key, "All_Gather", tp_degree, tensor_size)

        #     # Reset
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()

        #     if self.rank == 0:
        #         offline_data_manager._write_one_entry(query_key, all_gather_latency_ms, None)

        if self.rank == 0:
            offline_data_manager._store()

        # return {
        #     "all_reduce_latency_ms": all_reduce_latency * 1000,
        #     "all_gather_latency_ms": all_gather_latency * 1000
        # }



if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    nodes_num = (world_size - 1) // local_world_size + 1

    profiler = CommunicationOfflineProfiler(nodes_num, world_size, rank)

    profiler.run_profile(nodes_num, world_size)

    # for tensor_size in tensor_sizes:
    #     results = profiler.run_profile(tensor_sizes)

    #     if rank == 0:
    #         # print(f"Results for tensor size {tensor_size}: {results}")
    #         pass

    profiler.cleanup()