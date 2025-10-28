#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Main entrypoint of Crius Torch-like parallelism tuner. 

This module is independent to the underlying training backend (e.g., Megatron).
"""

from typing import (Any, Tuple, Optional, List, Callable, Dict)

import torch
from torch import nn

from crius.core.grid_estimator.planner import CriusParallelPlanner
from crius.core.profiler import CriusDistProfiler
from crius.core.grid_estimator.grid import Grid

__all__ = [ "CriusParallelTuner", ]


class CriusParallelTuner:
    """
    Crius Torch-like grid-based parallelism tuner.

    Roadmap of this module: (1) Enumerates all grids. Each grid includes all parallelism plans with the same pipeline 
    degree and resources; (2) Generate a Pareto-optimal set within each grid; (3) Enable disaggregated profiler 
    for each Pareto-optimal parallelism plan; (4) Select the optimal grid and its optimal parallelism plan.

    Args:
        num_nodes: Number of nodes.
        num_devices_per_node: Number of devices per node.
        grid_plan_optimization_mode: How we optimize parallelism plans in a grid. Options: ["all", ]. 
            If "all", in each grid, we enumerate and profile all Pareto-optimal plans and select the optimal one.
        profiled_data_path: Path of the offline profiled performance data.
    """

    def __init__(
        self,
        model: nn.Module,
        num_nodes: int,
        num_devices_per_node: int,
        grid_plan_optimization_mode: str = "all",
        profiled_data_path: Optional[str] = None,
    ):
        self.model = model
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.grid_plan_optimization_mode = grid_plan_optimization_mode

        self.planner = CriusParallelPlanner(model, pipeline_stage_partition_method="uniform")
        self.profiler = CriusDistProfiler(profiled_data_path)
    
    def tune(self):
        """ Tune the parallelism plan. """
        # Step 1. Enumerate grids
        grids: List[Grid] = self._enumerate_grids()
        # Step 2. Generate plans
        for grid in grids:
            self.planner.generate_pareto_optimal_plans(grid)
        # Step 3. Profile
        for grid in grids:
            if self.grid_plan_optimization_mode == "all":
                for plan in grid.pareto_optimal_plans:
                    raise NotImplementedError
            else:
                raise ValueError(f"Invalid grid plan optimization mode: {self.grid_plan_optimization_mode}")    
    
    def _enumerate_grids(self, force_power_of_two: bool = True):
        """ Enumerate all candidate grids. 
        
        Args:
            force_power_of_two: Force pipeline degree to be power-of-2.
        """

        grids = []
        pp_degree = 1
        while pp_degree <= self.num_nodes * self.num_devices_per_node:
            grids.append(
                Grid(self.num_nodes, self.num_devices_per_node, pp_degree)
            )
            
            if force_power_of_two:
                pp_degree *= 2
            else:
                raise ValueError(f"Currently only support PP degree to be power-of-2.")

        return grids
