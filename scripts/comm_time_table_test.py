import torch
import torch.distributed as dist
import time
import os
import sys
import re

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np
import pickle

import torch
from torch import nn



def _get_comm_time_table(data_dict) -> Dict[str, Dict[int, torch.dtype]]:
    """ Match target patterns from the loaded content. """
    pattern = r'^(all_reduce|all_gather|all_to_all|reduce_scatter)_n(\d+)_gs(\d+)_d(\d+)$'
    comm_time_table = {}

    

    for key, vdict in data_dict.items():
        match = re.match(pattern, key)
        if match:
            op_type = match.group(1)       
            n  = int(match.group(2))         
            gs = int(match.group(3))        
            d  = int(match.group(4))         
        else:
            raise ValueError(f"Unable to parse key: {key}")  
            
        _key = f'{op_type}_n{n}_gs{gs}'
        # print(key)
        # print(vdict)
        if _key not in comm_time_table:
            comm_time_table[_key] = []
        else:
            comm_time_table[_key].append([d, vdict['latency_ms']])
            
    print(comm_time_table)                 
    return comm_time_table

profiled_data_path = './crius/offline/prof_comm_data.pkl'

with open(profiled_data_path, 'rb') as f:
    profiled_data = pickle.load(f)

# print(profiled_data)

comm_data_table = _get_comm_time_table(profiled_data)

print(comm_data_table)