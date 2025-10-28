#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np

import torch
from torch import nn

from crius.core.grid_estimator.grid import Grid
from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (compile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode

__all__ = [ "CriusParallelPlanner", ]


class CriusParallelPlanner:
    """ Crius Torch-like intra-grid parallelism planner. 

    Roadmap of this module: (1) TODO
    
    Args:
        model: The model to be optimized.
        pipeline_stage_partition_method: How to partition operators into stages. 
            Options: ["uniform", "auto", ]. If "uniform", uniformly slice model operators 
            into stages. If "auto", automatically slice model operators based on FLOPs information.
    """

    def __init__(self, model: nn.Module, pipeline_stage_partition_method: bool):
        self.model = model
        self.pipeline_stage_partition_method = pipeline_stage_partition_method

    def _precompile_model(self):
        """ Pre-compile the model into IR graphs. """
        # TODO(chunyu): Can we do something like: pre-compile the model once, adjust the IR graphs for 
        #               each parallelism plan? We leave this for future implementation.
        raise NotImplementedError

    def generate_pareto_optimal_plans(self, grid: Grid) -> List[ParallelismPlan]:
        """ Generate Pareto-optimal set of parallelism plans in this grid. """

        # Enumerate parallelism plans

        # For each plan, compile the model
        
        # Get operator information: FLOPs, memory access
        
        # Assign per-(transformer) layer GPU quota

        # Determine pipeline stage boundary

        # Enumerate and formulate a Pareto-optimal plan set

    def _enumerate_candidate_parallelism_plans(self):
        """ Enumerate plans (`ParallelismPlan`). """
        raise NotImplementedError

    def _compile_model_with_one_plan(self, parallelism_plan: ParallelismPlan):
        """ Compile model into IR graphs based on the plan. """
        raise NotImplementedError
    
    def _get_model_operator_information(self):
        """ Get operator information of the model such as FLOPs and memory access. """
        raise NotImplementedError
    
    def _assign_per_layer_gpu_quota(self):
        """ TODO(chunyu): Which granularity should we use to assign? Operator or transformer layer? """
        raise NotImplementedError
    
    def _partition_pipeline_stages(self):
        """ TODO(chunyu): We should move `partition_pipeline_stages()` in `compiler.py` to here. """
        raise NotImplementedError
    
    def _formulate_pareto_optimal_plan_set(self):
        """ Formulate a set of Pareto-optimal parallelism plans. """
        raise NotImplementedError
