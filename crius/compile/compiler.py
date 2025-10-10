#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Implementation of Torch-like model compiler. """

from typing import (Any, List, Tuple, Callable, Optional, Dict)
from dataclasses import dataclass

import torch
from torch import nn
from torch.fx import GraphModule

from crius.compile.graph.transform import (
    transform_to_ir_graph, propagate_node_shapes, infer_node_layer_types, insert_comm_primitives_into_graph,
    construct_backward_graph)
from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan

__all__ = [
    "CompileConfigs",
    "compile",
]


@dataclass
class CompileConfigs:
    """ Compilation configurations. """
    
    base_input_args: Tuple[Any]
    """ Input arguments of model `.forward()` for IR graph transformation. """

    base_input_kwargs: Dict[str, Any]
    """ Input keyword arguments of model `.forward()` for IR graph transformation. """

    parallelism_plan: ParallelismPlan
    """ Parallelism plan of the model. """


def compile(
    model: nn.Module,
    compile_configs: CompileConfigs,
) -> IRGraph:
    """ Python-to-IR compilation. 
    
    Args:
        model: Torch-like model to be compiled.
        compile_configs: Compilation configurations.
    """
    
    # Step 1. Transform model into IR graph
    graph = transform_to_ir_graph(model, *compile_configs.base_input_args, **compile_configs.base_input_kwargs)
    # Step 2. Propagate per-node input/output shapes
    graph = propagate_node_shapes(graph, *compile_configs.base_input_args, **compile_configs.base_input_kwargs)
    graph = infer_node_layer_types(graph)
    # Step 3. Insert communication primitives
    graph = insert_comm_primitives_into_graph(graph, compile_configs.parallelism_plan, forward_pass=True)
    # Step 4. Construct backward graph
    bw_graph = construct_backward_graph(graph)
    bw_graph = insert_comm_primitives_into_graph(bw_graph, compile_configs.parallelism_plan, forward_pass=False)
    # Step 5. Pipeline stage partition
    partition_pipeline_stages(graph, compile_configs.parallelism_plan, bw_graph)

    return graph, bw_graph


def partition_pipeline_stages(
    graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: Optional[IRGraph] = None,
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

        elif parallelism_plan.pipeline_num_preprocess_nodes >=0 and parallelism_plan.pipeline_num_postprocess_nodes >=0:
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

        num_bw_nodes = len(list(bw_graph.nodes))
        assert num_bw_nodes % parallelism_plan.pipeline_parallel_degree == 0
        num_bw_nodes_per_stage = [
            num_bw_nodes // parallelism_plan.pipeline_parallel_degree for _ in range(parallelism_plan.pipeline_parallel_degree)
        ]
        __annotate_node_stage_index(num_bw_nodes_per_stage, bw_graph)
    
    elif parallelism_plan.pipeline_stage_partition_method == "manual":
        raise NotImplementedError

    else:
        raise ValueError(f"Invalid pipeline stage partition method: {parallelism_plan.pipeline_stage_partition_method}")
