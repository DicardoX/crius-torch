#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set
import unittest
import logging
from functools import partial
from collections import OrderedDict
import time
import numpy as np
import torch
from torch import nn
import argparse
import gc
import unittest




class TorchModelProfileTest(unittest.TestCase):
    """ Basic torch model disaggregated profiling test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """
    # @unittest.skip("Pass.")
    def test_torch_model_offline_profiling1(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=1,
            max_batch_size=4,
            batch_size_step="power-of-2",
            min_seq_len=32,
            max_seq_len=2096,
            seq_len_step="power-of-2",
            gemm_op_type=torch.mul,
            gemm_left_op_hidden_dims=[1024, 2048, 4096, ],
            gemm_right_op_hidden_dims=[1024, 2048, 4096, ],
            dtype=torch.float16,
            device="cuda",
            num_warmup_steps=5,
            num_profile_steps=10,
            force_overwrite=True,
        )

        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        
        print("\ntorch.mul ===========================")
        offline_profiler.benchmark_layernorm(8, 4096, 2048)
        return 
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.tril
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch tril ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.eq
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch eq ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.add
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch add ===========================")
        offline_profiler.profile_single_input_gemm()

        profile_configs.gemm_op_type = torch.addmm
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        print("\n\ntorch addmm ===========================")
        offline_profiler.benchmark_addmm(1600, 6400, 4096)

    # @unittest.skip("Pass.")
    def test_torch_model_offline_profiling(self):
        """ Offline profile all operators."""

        from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
        profile_configs = OfflineProfileConfigs(
            min_batch_size=1,
            max_batch_size=4,
            batch_size_step="power-of-2",
            min_seq_len=512,
            max_seq_len=4096,
            seq_len_step="power-of-2",
            gemm_left_op_hidden_dims=[1024, 2048, 4096, ],
            gemm_right_op_hidden_dims=[1024, 2048, 4096, ],
            dtype=torch.float16,
            device="cuda:0",
            num_warmup_steps=5,
            num_profile_steps=10,
            force_overwrite=True,
        )
        offline_profiler = CriusOfflineDataManager(
            profile_configs=profile_configs,
            to_store_path='./crius/offline/',
            data_filename='prof_data.pkl',
        )
        
        offline_profiler.profile_all_gemm()


if __name__ == '__main__':
    unittest.main()
