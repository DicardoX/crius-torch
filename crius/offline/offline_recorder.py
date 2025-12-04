from typing import overload
import os
import pickle
import gc 
from crius.global_envs import logger 

__all__ = [ "OfflineDataRecorder", ]


class OfflineDataRecorder:
    """
    OfflineDataRecorder class that writes "*.pkl" file about computation and communication operators profiling entry.

    Args:
        to_store_path: str: The path where data is stored.
        data_filename: The name of "*.pkl" file.
        force_overwrite: Whether to overwrite the value when the key exists.
    """

    def __init__(
        self,
        to_store_path: str = "./crius/offline/data",
        data_filename: str = "prof_data.pkl",
        force_overwrite: bool = True
    ):
        """ Initialize a OfflineDataRecorder object. """

        self.to_store_path = to_store_path
        self.filename = data_filename
        self._path = os.path.join(self.to_store_path, self.filename)
        self.force_overwrite = force_overwrite
        if not os.path.exists(to_store_path):
            os.makedirs(os.path.dirname(to_store_path), exist_ok=True)
            logger.warning(f"Created directory for profiling data:{os.path.dirname(to_store_path)}")

        self.data = None
        self._load()

    def _load(self) -> None:
        """ Load the collected profiling data from the specified path. """        
        if not os.path.exists(self._path):
            logger.warning(f"No profiling data found at {self.to_store_path}. Starting from-scratch profiling.")
            self.data = {}
            return
        
        if self.data is None:
            with open(self._path, 'rb') as f:
                self.data = pickle.load(f)
    
    def write_one_entry(self, key: str, latency_ms: float, memory_footprint_mb: float,) -> None:
        """ Write a single profiling entry to the data store. """
        if key in self.data and not self.force_overwrite:
            raise ValueError(f"Key {key} already exists in the profiling data.")
        
        self.data[key] = {
            "latency_ms": latency_ms,
            "memory_footprint_mb": memory_footprint_mb,
        }

    def store(self) -> None:
        """ Store the collected profiling data to the specified path. """
        with open(self._path, 'wb') as f:
            pickle.dump(self.data, f)
        logger.info(f"\n\n=========> Offline profiling data is stored in: {self.to_store_path}") 
    
    @classmethod
    def get_query_key(
        cls, 
        op_type: str, 
        batch_size: int,
        seq_len: int, 
        left_op_hidden_dim: int,
        right_op_hidden_dim: int, 
        layout: str,
        comm_node_num: int = 1,
        comm_group_size: int = 1, 
        comm_data_size: int = 1
    ) -> str:
        """ Generate a unique query key for a computation or communication operation. """

        if   op_type == "gemm":
            return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_r{right_op_hidden_dim}_layout_[{layout}]"
        elif op_type == "bmm":
            assert right_op_hidden_dim is None
            return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_layout_[{layout}]"
        elif op_type == "reduce_sum":
            assert seq_len is None
            return f"{op_type}_b{batch_size}_l{left_op_hidden_dim}_r{right_op_hidden_dim}_layout_[{layout}]"
        elif op_type == "baddbmm":
            assert right_op_hidden_dim is None
            return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_layout_[{layout}]"
        elif op_type in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
            assert comm_group_size is not None, f"Communication group size ({comm_group_size}) is not set."
            return f"{op_type}_n{comm_node_num}_gs{comm_group_size}_d{comm_data_size}"
        
    @overload
    def _check_cache(
        self, 
        query_key: str, 
        op_type: str, 
        batch_size: int, 
        seq_len: int, 
        left_hidden_dim: int, 
        right_hidden_dim: int,
        layout: str
    ) -> None: ...

    @overload
    def _check_cache(
        self, 
        query_key: str, 
        op_type: str, 
        comm_node_num: int,
        group_size: int, 
        data_size: int
    ) -> None: ...    

    def _check_cache(self, query_key, op_type, *args, **kwargs) -> None: 
        """ Check whether query_key is in '*.pkl' file. """
        if len(args) == 5:   

            batch_size, seq_len, left_hidden_dim, right_hidden_dim, layout = args  

            if query_key in self.data and not self.force_overwrite:
                logger.info(f"Cached {op_type} profile: batch_size: {batch_size}, seq_len: {seq_len}, " + 
                                        f"left_hd: {left_hidden_dim}, right_hd: {right_hidden_dim}, layout: {layout}")
                return
                
            logger.info(f"=========> {op_type} profile: batch_size: {batch_size}, seq_len: {seq_len}, " + 
                                        f"left_hd: {left_hidden_dim}, right_hd: {right_hidden_dim}, layout: {layout}")
                 
        elif len(args) == 3:       

            comm_node_num, group_size, data_size = args
            
            if query_key in self.data and not self.force_overwrite:

                logger.info(f"Cached {op_type} profile: comm_node_num: {comm_node_num}, group_size: {group_size}, data_size: {data_size}")
                return
                
            logger.info(f"=========> {op_type} profile: comm_node_num: {comm_node_num}, group_size: {group_size}, data_size: {data_size}")
