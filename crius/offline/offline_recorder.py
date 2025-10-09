from typing import overload
import os
import pickle
import gc 
from crius.global_envs import logger 


class OfflineDataRecorder:
    def __init__(
        self,
        to_store_path: str = "./crius/offline",
        data_filename: str = "prof_data.pkl",
        force_overwrite: bool = True
    ):
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

    @overload
    def _check_cache(
        self, 
        query_key: str, 
        op_type: str, 
        batch_size: int, 
        seq_len: int, 
        left_hidden_dim: int, 
        right_hidden_dim: int
    ) -> None: ...

    @overload
    def _check_cache(
        self, 
        query_key: str, 
        op_type: str, 
        tp_degree: int, 
        data_size: int
    ) -> None: ...    

    def _check_cache(self, query_key, op_type, *args, **kwargs) -> None: 
        if len(args) == 4:   

            batch_size, seq_len, left_hidden_dim, right_hidden_dim = args  

            if query_key in self.data and not self.force_overwrite:
                logger.info(f"Cached GEMM profile: batch_size: {batch_size}, seq_len: {seq_len}, " + 
                                        f"left_hd: {left_hidden_dim}, right_hd: {right_hidden_dim}")
                return
                
            logger.info(f"=========> GEMM profile: batch_size: {batch_size}, seq_len: {seq_len}, " + 
                                    f"left_hd: {left_hidden_dim}, right_hd: {right_hidden_dim}")
                 
        elif len(args) == 2:       
            if query_key in self.data and not self.force_overwrite:
                
                tp_degree, data_size = args

                logger.info(f"Cached {op_type} profile: tp_degree: {tp_degree}, data_size: {data_size}")
                return
                
            logger.info(f"=========> {op_type} profile: tp_degree: {tp_degree}, data_size: {data_size}")    
    
    def _write_one_entry(self, key: str, latency_ms: float, memory_footprint_mb: float,) -> None:
        """ Write a single profiling entry to the data store. """
        if key in self.data and not self.force_overwrite:
            raise ValueError(f"Key {key} already exists in the profiling data.")
        
        self.data[key] = {
            "latency_ms": latency_ms,
            "memory_footprint_mb": memory_footprint_mb,
        }

    def _store(self) -> None:
        """ Store the collected profiling data to the specified path. """
        with open(self._path, 'wb') as f:
            pickle.dump(self.data, f)
        logger.info(f"\n\n=========> Offline profiling data is stored in: {self.to_store_path}")

    @overload
    @classmethod
    def get_query_key(
        cls, 
        op_type: str, 
        batch_size: int, 
        seq_len: int, 
        left_op_hidden_dim: int, 
        right_op_hidden_dim: int
    ) -> str: ...  

    @overload
    @classmethod
    def get_query_key(
        cls, 
        op_type: str, 
        tp_degree: int, 
        data_size: int
    ) -> str: ...  
    
    @classmethod
    def get_query_key(
        cls, 
        op_type: str, 
        *args,
        **kwargs
    ) -> str:
        """ Generate a unique query key for the GEMM operation. """
        # return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_r{right_op_hidden_dim}"
        if len(args) == 4:
            batch_size, seq_len, left_op_hidden_dim, right_op_hidden_dim = args
            return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_r{right_op_hidden_dim}"
        elif len(args) == 2:
            tp_degree, data_size = args
            return f"{op_type}_tp{tp_degree}_ds{data_size}"