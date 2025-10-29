#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np
from copy import deepcopy

import torch
from torch import nn

from crius.core.grid_estimator.grid import Grid
from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (postcompile, CompileConfigs)
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

    def __init__(
        self, 
        model: nn.Module, 
        proto_parallelism_plan: ParallelismPlan,
    ):
        self.model = model
        self._proto_plan = proto_parallelism_plan

        self._power_of_two_degree = True
        """ Force all degrees to be power-of-2. """

    def generate_pareto_optimal_plans(self, grid: Grid, graph: IRGraph) -> List[ParallelismPlan]:
        """ Generate Pareto-optimal set of parallelism plans in this grid. 
        
        TODO(chunyu): This CPU-only process can be accelerated via multi-thread.
        """

        # Enumerate parallelism plans
        plans = self._enumerate_candidate_parallelism_plans(grid)

        # For each plan, compile the model
        for plan in plans:
            # Parallelism-specific post-compile
            graph, bw_graph = postcompile(graph, plan)
            # Get operator information: FLOPs, memory access
            self._get_model_operator_information(graph, bw_graph)
            # Assign per-(transformer) layer GPU quota

            # Determine pipeline stage boundary
        
        exit(0)

        # Enumerate and formulate a Pareto-optimal plan set

    def _enumerate_candidate_parallelism_plans(self, grid: Grid):
        """ Enumerate plans (`ParallelismPlan`). """
        
        num_nodes, num_devices_per_node, pp = grid.num_nodes, grid.num_devices_per_node, grid.pipeline_degree
        num_gpus = num_nodes * num_devices_per_node
        plans = []
        
        tp = 1
        while tp <= num_gpus:
            assert num_gpus % tp == 0
            dp = num_gpus // tp
            plans.append(
                ParallelismPlan(
                    self._proto_plan.pipeline_type,
                    pipeline_parallel_degree=pp,
                    num_layers=self._proto_plan.num_layers,
                    num_microbatches=self._proto_plan.num_microbatches,
                    tensor_parallel_degree=tp,
                    data_parallel_degree=dp,
                    pipeline_stage_partition_method=self._proto_plan.pipeline_stage_partition_method,
                    pipeline_use_boundary_markers=self._proto_plan.pipeline_use_boundary_markers,
                    pipeline_boundary_start_marker=self._proto_plan.pipeline_boundary_start_marker,
                    pipeline_boundary_end_marker=self._proto_plan.pipeline_boundary_end_marker,
                )
            )

            if self._power_of_two_degree:
                tp *= 2
            else:
                raise ValueError(f"Currently only supports power-of-2 degree in plan enumeration.")
        
        return plans
    
    def _get_model_operator_information(self, graph: IRGraph, bw_graph: Optional[IRGraph] = None):
        """ Get operator information of the model such as FLOPs and memory access. """

        # TODO(chunyu): Given an IR graph, how we get FLOPs and memory per operator?
        
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
