#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np
from copy import deepcopy

import torch
from torch import nn

from crius.core.grid_planner.grid import Grid
from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (postcompile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode

__all__ = [ "CriusParallelPlanner", ]


class CriusParallelPlanner:
    """ Crius Torch-like intra-grid parallelism planner. 

    Roadmap of this module: (1) TODO
    
    Args:
        model: The model to be optimized.
        proto_parallelism_plan: A prototype of parallelism plan that includes some necessary attributes.
    """

    def __init__(
        self, 
        model: nn.Module, 
        proto_parallelism_plan: ParallelismPlan,
    ):
        self.model = model
        self._proto_plan = proto_parallelism_plan
        self.pipeline_stage_partition_method = proto_parallelism_plan.pipeline_stage_partition_method
        """ How to partition operators into stages. Options: ["uniform", "auto", ]. If "uniform",
        uniformly slice model operators into stages. If "auto", automatically slice model operators 
        based on FLOPs information. """

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

            if self.pipeline_stage_partition_method == "uniform": 
                pass    # No information needed
            elif self.pipeline_stage_partition_method == "auto":
                self._get_model_operator_information(graph, bw_graph)
                self._assign_per_layer_gpu_quota()
            else:
                raise ValueError(f"Invalid pipeline stage partition method: {self.pipeline_stage_partition_method}")

            # Determine pipeline stage boundary
            self._partition_pipeline_stages(graph, plan, bw_graph)
        
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
    
    def _partition_pipeline_stages(
        self, graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: Optional[IRGraph] = None,
    ) -> None:
        """ Annotate all nodes' pipeline stage index. """

        def __annotate_node_stage_index(num_nodes_per_stage_: List[int], graph_):
            this_stage_node_num, cur_stage_index = 0, 0
            for node in graph_.nodes:
                this_stage_node_num += 1
                if this_stage_node_num <= num_nodes_per_stage_[cur_stage_index]:
                    setattr(node, "stage_index", cur_stage_index)

                if this_stage_node_num == num_nodes_per_stage_[cur_stage_index]:
                    cur_stage_index += 1
                    this_stage_node_num = 0

        # Partition forward graph
        if parallelism_plan.pipeline_stage_partition_method == "uniform":
            # Uniformly partition looped layers (e.g., transformer layers)
            if parallelism_plan.pipeline_use_boundary_markers:
                # Manually specify the boundary markers to indicate the start and end of a layer
                assert parallelism_plan.pipeline_boundary_start_marker is not None and parallelism_plan.pipeline_boundary_end_marker is not None, \
                    f"When using boundary markers to partition pipeline layers, users must specify the start and end markers."
                num_fw_nodes_per_layer = [0 for _ in range(parallelism_plan.num_layers)]
                cur_layer_index, num_marked_nodes = -1, 0
                for node in graph.nodes:
                    if any([_marker in str(node.target) for _marker in parallelism_plan.pipeline_boundary_start_marker]):
                        num_marked_nodes += 1
                    else:
                        num_marked_nodes = 0
                    
                    if num_marked_nodes == len(parallelism_plan.pipeline_boundary_start_marker):
                        cur_layer_index += 1
                        num_marked_nodes = 0
                    
                    num_fw_nodes_per_layer[max(cur_layer_index, 0)] += 1

                assert parallelism_plan.num_layers % parallelism_plan.pipeline_parallel_degree == 0, \
                    f"The number of layers ({parallelism_plan.num_layers}) must be divisible by the pipeline parallel degree ({parallelism_plan.pipeline_parallel_degree})."
                layers_per_stage = parallelism_plan.num_layers // parallelism_plan.pipeline_parallel_degree
                num_fw_nodes_per_stage = []
                for i in range(parallelism_plan.pipeline_parallel_degree):
                    num_fw_nodes_per_stage.append(sum(num_fw_nodes_per_layer[i*layers_per_stage:(i+1)*layers_per_stage]))   
                __annotate_node_stage_index(num_fw_nodes_per_stage, graph)       

            elif parallelism_plan.pipeline_num_preprocess_nodes >= 0 and parallelism_plan.pipeline_num_postprocess_nodes >= 0:
                # Manually specify the number of preprocess/postprocess nodes before looped stages
                # assert parallelism_plan.pipeline_num_preprocess_nodes >=0 and parallelism_plan.pipeline_num_postprocess_nodes >= 0, \
                #     f"In uniform pipeline partition mode, users must specify the number of pre/post-process nodes."
                num_fw_nodes = len(list(graph.nodes))
                num_fw_loop_nodes = (num_fw_nodes - parallelism_plan.pipeline_num_preprocess_nodes - parallelism_plan.pipeline_num_postprocess_nodes)
                assert num_fw_loop_nodes % parallelism_plan.pipeline_parallel_degree == 0
                num_fw_loop_nodes_per_stage_ = num_fw_loop_nodes // parallelism_plan.pipeline_parallel_degree
                num_fw_nodes_per_stage = [num_fw_loop_nodes_per_stage_ for _ in range(parallelism_plan.pipeline_parallel_degree)]
                num_fw_nodes_per_stage[0] += parallelism_plan.pipeline_num_preprocess_nodes
                num_fw_nodes_per_stage[-1] += parallelism_plan.pipeline_num_postprocess_nodes
                __annotate_node_stage_index(num_fw_nodes_per_stage, graph)
            
            else:
                raise ValueError(f"In uniform pipeline partition mode, users must either specify the boundary markers or the number of pre/post-process nodes.")

            if bw_graph is None:
                return
        
        elif parallelism_plan.pipeline_stage_partition_method == "manual":
            raise NotImplementedError

        else:
            raise ValueError(f"Invalid pipeline stage partition method: {parallelism_plan.pipeline_stage_partition_method}")
        
        # Partition backward graph
        for node in bw_graph.nodes:
            assert node.fw_node.stage_index is not None, f"Stage index of forward node '{node.fw_node.target}' is not properly set."
            # bw_stage_index = 2 * parallelism_plan.pipeline_parallel_degree - node.fw_node.stage_index - 1
            setattr(node, "stage_index", node.fw_node.stage_index)
    
    def _formulate_pareto_optimal_plan_set(self):
        """ Formulate a set of Pareto-optimal parallelism plans. """
        raise NotImplementedError
