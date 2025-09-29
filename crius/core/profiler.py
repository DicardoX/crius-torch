#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Main entrypoint of Crius Torch-like profiler. 

This module is independent to the underlying training backend (e.g., Megatron).
"""

from typing import (Any, Tuple, Optional, List)
import numpy as np
import pickle

import torch
from torch import nn

from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (compile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode
from crius.offline.offline_profiler import CriusOfflineDataManager
from crius.global_envs import logger


class CriusDistProfiler:
    """ Crius Torch-like disaggregated profiler. 
    
    This module provides (1) operator disaggregation and (2) Device-free model profiling.
    """

    def __init__(self, profiled_data_path: Optional[str] = None) -> None:
        self._profiled_data = self._load_profiled_data(profiled_data_path)

    def _load_profiled_data(self, profiled_data_path: Optional[str] = None) -> None:
        """ Load profiled data from the given path. 
        
        This function is used to load offline profiled data of operator latencies.
        """
        if profiled_data_path is None:
            logger.warning(f"No profiled data path provided, using latencies of all 1.0ms for debug.\n\n")
            return None
        
        with open(profiled_data_path, 'rb') as f:
            profiled_data = pickle.load(f)
        return profiled_data

    def profile(
        self, model: nn.Module, 
        base_input_args: Tuple[Any],
        parallelism_plan: ParallelismPlan,
        only_forward: bool = False,
    ) -> Optional[float]:
        """ Profile the model. 
        
        Args:
            model: Torch-like model to be profiled.
            base_input_args: Input arguments of model `.forward()` for IR graph transformation.
            parallelism_plan: Parallelism plan of the model.
            only_forward: If True, only profile the forward pass. Default is False.
        
        Returns:
            e2e_latency: End-to-end pipeline latency of the model in milliseconds, or None
        """

        # Step 1. Compile the model into IR graph
        compile_configs = CompileConfigs(
            base_input_args=base_input_args,
            parallelism_plan=parallelism_plan,
        )
        graph = compile(model, compile_configs)

        # Step 2. Query and record per-operator latencies
        graph = self._query_operator_latencies_ms(graph)
        for node in graph.nodes:
            logger.debug(f"=====> Node target: {node.target}")
            logger.debug(f"Node name: {node.name} | OP type: {node.op} | " + 
                         f"Next: {node.next} | Input shapes: {node.input_shapes} | " + 
                         f"Output shapes: {node.output_shapes} | Latency: {node.latency}\n")

        # Step 3. Compute per-stage latency
        stage_latencies = self._compute_stage_latencies(graph, parallelism_plan)  

        # Step 4. Model end-to-end pipeline latency
        e2e_latency = self._model_e2e_pipeline_latency_with_oom_check(
            stage_latencies, parallelism_plan, only_forward,
        )
        return e2e_latency

    def _query_operator_latencies_ms(self, graph: IRGraph):
        """ Query operator latencies(ms) from offline profiled data. """

        for node in graph.nodes:
            if str(node.op) in ("call_module", "call_function", "call_method"):
                node_latency = 0.0
                if self._profiled_data is not None:
                    # Query the latency from the profiled data
                    input_shapes, output_shapes = node.input_shapes, node.output_shapes
                    query_key = CriusOfflineDataManager.get_query_key(
                        op_type=node.layer_type, batch_size=input_shapes[0][0], seq_len=input_shapes[0][1],
                        left_op_hidden_dim=input_shapes[0][2], right_op_hidden_dim=output_shapes[0][2],
                    )
                    if query_key in self._profiled_data:
                        node_latency = self._profiled_data[query_key]["latency_ms"]
                    else:
                        logger.warning(f"Query key {query_key} not found in profiled data, using default latency {node_latency}ms.")
                
                setattr(node, "latency", node_latency)
            else:
                # Non dominant nodes
                setattr(node, "latency", None)

        return graph

    def _compute_stage_latencies(self, graph: IRGraph, parallelism_plan: ParallelismPlan) -> List[List[float]]:
        """ Compute per-stage latencies. """

        stage_latencies = [[0.0, 0.0] for _ in range(parallelism_plan.pipeline_parallel_degree)]
        for node in graph.nodes:
            layer_index = parallelism_plan.op_to_layer_mapping.get(node.target, None)
            if layer_index is None:
                continue
            stage_index = next(
                (i for i, stage in enumerate(parallelism_plan.layer_to_stage_mapping) if layer_index in stage), None
            )
            # Accumulate latency
            if node.latency is not None and stage_index is not None:
                stage_latencies[stage_index][0] += node.latency
        
        # For backward pass, we assume the latency is 2x of the forward pass.
        for i in range(parallelism_plan.pipeline_parallel_degree):
            stage_latencies[i][1] = stage_latencies[i][0] * 2.0

        return stage_latencies

    def _model_e2e_pipeline_latency_with_oom_check(
        self, stage_latencies: List[List[float]], parallelism_plan: ParallelismPlan, only_forward: bool = False,
    ) -> Optional[float]:
        """ Compute model end-to-end pipeline latency with OOM check. """
        
        if parallelism_plan.pipeline_type == "1f1b":
            return self._model_e2e_pipeline_latency_with_oom_check_1f1b_schedule(
                stage_latencies, parallelism_plan, only_forward,
            )
        else:
            raise ValueError(f"Pipeline type {parallelism_plan.pipeline_type} is not supported yet.")
    
    def _model_e2e_pipeline_latency_with_oom_check_1f1b_schedule(
        self, stage_latencies: List[float], parallelism_plan: ParallelismPlan, only_forward: bool = False,
    ) -> Optional[float]:
        """ 1F1B pipeline schedule. 

                   Warmup |         Steady          | Drain
        clock num    0    |   1     2     3     4   |   5
        ---------  -----  | ----- ----- ----- ----- | -----
         stage 0   (0,0)  | (0,1)       (0,2)       | (0,3)
         stage 1          | (1,0) (1,2) (1,1) (1,3) |

        where `(i,j)` denotes stage `i` executes micro-batch `j` when `j < #micro-batches`,
        and denotes stage `i` executes micro-batch `j - #micro-batches` when `j >= #micro-batches`.

        Note that the backward pass's latency is 2x of the forward pass.
        """
        
        num_microbatches = parallelism_plan.num_microbatches
        num_stages = parallelism_plan.pipeline_parallel_degree

        if only_forward:
            # If only forward pass is profiled, we only consider the steady phase.
            fw_lats = [_l[0] for _l in stage_latencies]
            t_fw = np.max(fw_lats) * (num_microbatches - 1) + np.sum(fw_lats)
            return t_fw

        # Phase 1. Warmup
        t_warmup = np.sum([stage_latencies[i][0] for i in range(num_stages - 1)])

        # Phase 2. Steady
        # NOTE: Here, we assume perfect load balancing across pipeline stages. Therefore, we use 
        #       the last stage's latency refer to DynaPipe's Figure 10.
        t_steady = num_microbatches * (stage_latencies[-1][0] + stage_latencies[-1][1])

        # Phase 3. Drain
        t_drain = np.sum([stage_latencies[i][1] for i in range(num_stages - 1)])

        return (t_warmup + t_steady + t_drain)
