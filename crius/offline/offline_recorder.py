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
        to_store_path: str = "./crius/offline",
        data_filename: str = "prof_data.pkl",
        force_overwrite: bool = True,
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
        num_attn_heads: int = 1,
        num_comm_nodes: int = 1,
        comm_group_size: int = 1, 
        comm_data_size: int = 1,
        attn_head_split: bool = True,
    ) -> str:
        """ Generate a unique query key for a computation or communication operation. 
        
        Args:
            attn_head_split (bool): Split attention heads and accumulate num_attn_heads to batch dimension. 
                Set to false if the attention head has already been splitted (e.g., shapes got from IR graph).
        """

        if op_type == "gemm":
            if "*" not in layout:
                return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_r{right_op_hidden_dim}_layout[{layout}]"
            else:
                # Batch size and sequence length dimensions are merged (used in weight gradient computation)
                if batch_size > 1:
                    seq_len *= batch_size   # Accumulated
                return f"{op_type}_b*s{seq_len}_l{left_op_hidden_dim}_r{right_op_hidden_dim}_layout[{layout}]"
        elif op_type in ("bmm", "baddbmm"):
            assert right_op_hidden_dim is None
            if attn_head_split:
                batch_size *= num_attn_heads
                left_op_hidden_dim = left_op_hidden_dim // num_attn_heads
            return f"{op_type}_b{batch_size}_s{seq_len}_l{left_op_hidden_dim}_layout[{layout}]"
        elif op_type == "layernorm":
            return f"{op_type}_b{batch_size}_s{seq_len}_h{left_op_hidden_dim}_layout[{layout}]"
        elif op_type == "dropout":
            if left_op_hidden_dim is None:
                return f"{op_type}_b{batch_size}_s{seq_len}_n{num_attn_heads}_layout[{layout}]"
            elif num_attn_heads is None:
                return f"{op_type}_b{batch_size}_s{seq_len}_h{left_op_hidden_dim}_layout[{layout}]"
            else:
                raise ValueError("For Dropout op, either left_op_hidden_dim or num_attn_heads should be None.")
        elif op_type in ("masked_fill_", "softmax",):
            return f"{op_type}_b{batch_size}_s{seq_len}_n{num_attn_heads}_layout[{layout}]"
        elif op_type in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter",):
            assert comm_group_size is not None, f"Communication group size ({comm_group_size}) is not set."
            return f"{op_type}_n{num_comm_nodes}_gs{comm_group_size}_d{comm_data_size}"
        else:
            return None
    
    @classmethod
    def parse_query_key(cls, query_key: str) -> dict:
        """ Parse a query key into its components. """
        components = query_key.split('_')
        parsed = {}
        for comp in components:
            if comp.startswith('b'):
                parsed['batch_size'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('s'):
                parsed['seq_len'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('l'):
                parsed['left_op_hidden_dim'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('r'):
                parsed['right_op_hidden_dim'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('n'):
                parsed['num_attn_heads'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('h'):
                parsed['hidden_dim'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('gs'):
                parsed['comm_group_size'] = int(comp[2:]) if comp[2:] != 'None' else None
            elif comp.startswith('d'):
                parsed['comm_data_size'] = int(comp[1:]) if comp[1:] != 'None' else None
            elif comp.startswith('layout'):
                layout_str = comp.split('[')[1].rstrip(']') if 'None' not in comp else None
                parsed['layout'] = layout_str
            else:
                parsed['op_type'] = comp
        return parsed
