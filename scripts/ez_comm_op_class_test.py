import torch
import torch.distributed as dist
import time
import argparse
import signal
import os
from typing import List, Dict, Tuple

class CommunicationProfiler:
    def __init__(self, backend='nccl'):
        self.backend = backend
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._should_stop = False
        
    def setup(self):
        """Initialize communication environment"""
        dist.init_process_group(self.backend)
        
        # 正确处理GPU设备分配
        if torch.cuda.is_available():
            # 确保每个进程使用不同的GPU
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            
        if self.rank == 0:
            print(f"Initialized process group with {self.world_size} devices")
            if torch.cuda.is_available():
                print(f"Available GPUs: {torch.cuda.device_count()}")
    
    def cleanup(self):
        """Clean up communication environment"""
        dist.destroy_process_group()
    
    def _handle_signal(self, signum, frame):
        """Handle interrupt signals gracefully"""
        if self.rank == 0:
            print(f"\nReceived signal {signum}, shutting down...")
        self._should_stop = True
    
    def profile_all_comm(self, 
                        data_sizes: List[int], 
                        tp_sizes: List[int],
                        num_iterations: int = 100,
                        warmup_iterations: int = 10) -> Dict[str, List[float]]:
        """
        Profile all_reduce and all_gather operations with different data sizes and TP degrees
        """
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        def create_tp_group(tp_size: int):
            """Create TP communication group"""
            if self.world_size % tp_size != 0:
                return None
            
            group_id = self.rank // tp_size
            start_rank = group_id * tp_size
            ranks = list(range(start_rank, start_rank + tp_size))
            return dist.new_group(ranks)
        
        def profile_all_reduce(tensor_size, dtype, group):
            """Profile all_reduce operation"""
            # 使用当前设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor = torch.randn(tensor_size, dtype=dtype, device=device)
            
            # Warmup
            for _ in range(warmup_iterations):
                dist.all_reduce(tensor, group=group)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if self._should_stop:
                    return float('nan')
            
            # Timed iterations
            start_time = time.time()
            for _ in range(num_iterations):
                dist.all_reduce(tensor, group=group)
                if self._should_stop:
                    return float('nan')
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            return elapsed / num_iterations
        
        def profile_all_gather(tensor_size, dtype, group):
            """Profile all_gather operation"""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_tensor = torch.randn(tensor_size, dtype=dtype, device=device)
            group_size = dist.get_world_size(group)
            output_tensors = [torch.zeros_like(local_tensor) for _ in range(group_size)]
            
            # Warmup
            for _ in range(warmup_iterations):
                dist.all_gather(output_tensors, local_tensor, group=group)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if self._should_stop:
                    return float('nan')
            
            # Timed iterations
            start_time = time.time()
            for _ in range(num_iterations):
                dist.all_gather(output_tensors, local_tensor, group=group)
                if self._should_stop:
                    return float('nan')
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            return elapsed / num_iterations
        
        # Initialize results
        results = {
            'all_reduce_latencies': [],
            'all_gather_latencies': [],
            'configurations': []
        }
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            if self.rank == 0:
                print(f"Using CUDA with {torch.cuda.device_count()} GPUs")
        else:
            if self.rank == 0:
                print("Using CPU for communication")
        
        # Main test loop
        for tp_size in tp_sizes:
            if self._should_stop:
                break
                
            if self.world_size % tp_size != 0:
                if self.rank == 0:
                    print(f"Skipping TP size {tp_size} (not divisible by world size {self.world_size})")
                continue
                
            for data_size in data_sizes:
                if self._should_stop:
                    break
                    
                dtype = torch.float32
                element_size = torch.tensor([], dtype=dtype).element_size()
                tensor_size = (data_size,)
                data_size_mb = data_size * element_size / (1024 * 1024)
                
                tp_group = create_tp_group(tp_size)
                if tp_group is None:
                    continue
                
                # Profile operations
                ar_latency = profile_all_reduce(tensor_size, dtype, tp_group)
                ag_latency = profile_all_gather(tensor_size, dtype, tp_group)
                
                if self.rank == 0 and not self._should_stop:
                    config_str = f"TP{tp_size}_Data{data_size}elements({data_size_mb:.2f}MB)"
                    results['all_reduce_latencies'].append(ar_latency)
                    results['all_gather_latencies'].append(ag_latency)
                    results['configurations'].append(config_str)
                    
                    print(f"{config_str}: "
                          f"all_reduce={ar_latency*1000:.3f}ms, "
                          f"all_gather={ag_latency*1000:.3f}ms")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Profile collective communication operations')
    parser.add_argument('--backend', type=str, default='gloo', 
                       choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--data-sizes', type=int, nargs='+', 
                       default=[1024, 2048, 4096], 
                       help='Data sizes to test (number of elements)')
    parser.add_argument('--tp-sizes', type=int, nargs='+', 
                       default=[2, 4], 
                       help='TP sizes to test')
    parser.add_argument('--iterations', type=int, default=50, 
                       help='Number of iterations for profiling')
    parser.add_argument('--warmup', type=int, default=5, 
                       help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    dist.init_process_group(args.backend)
    profiler = CommunicationProfiler(args.backend)
    
    try:
        results = profiler.profile_all_comm(
            data_sizes=args.data_sizes,
            tp_sizes=args.tp_sizes,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
        
        if profiler.rank == 0:
            print("\n=== Final Results ===")
            for config, ar, ag in zip(results['configurations'], 
                                    results['all_reduce_latencies'],
                                    results['all_gather_latencies']):
                print(f"{config}:")
                print(f"  all_reduce: {ar*1000:.3f}ms ({1/ar:.1f} ops/sec)")
                print(f"  all_gather: {ag*1000:.3f}ms ({1/ag:.1f} ops/sec)")
                print()
                
    finally:
        profiler.cleanup()


if __name__ == "__main__":
    main()