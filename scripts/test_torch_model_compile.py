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

import torch
from torch import nn


class TorchModelCompileTest(unittest.TestCase):
    """ Basic torch model compilation test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    def test_torch_model_compilation(self):
        """ Compile a naive torch model into IR and parse its computation operators. """
        
        # TODO(chunyu): Implement the test case.
        raise NotImplementedError()


if __name__ == '__main__':
    unittest.main()
