#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Main entrypoint of Crius Torch-like profiler. 

This module is independent to the underlying training backend (e.g., Megatron).
"""

from typing import (Any, Tuple, Optional, List)

import torch
from torch import nn

from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (compile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode
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
        
        raise NotImplementedError()

    def profile(
        self, model: nn.Module, 
        base_input_args: Tuple[Any],
        parallelism_plan: ParallelismPlan,
    ) -> Optional[float]:
        """ Profile the model. 
        
        Args:
            model: Torch-like model to be profiled.
            base_input_args: Input arguments of model `.forward()` for IR graph transformation.
            parallelism_plan: Parallelism plan of the model.
        
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
            stage_latencies, parallelism_plan,
        )
        return e2e_latency

    def _query_operator_latencies_ms(self, graph: IRGraph):
        """ Query operator latencies(ms) from offline profiled data. """

        for node in graph.nodes:
            if node.op == "call_module":
                node_latency = 1.0
                if self._profiled_data is not None:
                    # Query the latency from the profiled data
                    raise NotImplementedError()
                
                setattr(node, "latency", node_latency)
            else:
                setattr(node, "latency", None)

        return graph

    def _compute_stage_latencies(self, graph: IRGraph, parallelism_plan: ParallelismPlan) -> None:
        """ Compute per-stage latencies. 
        
        TODO(chunyu): Currently only supports forward pass, backward pass can be estimated by 2x forward pass. 
        """

        stage_latencies = [0.0] * parallelism_plan.pipeline_parallel_degree
        for node in graph.nodes:
            raise NotImplementedError()

    def _model_e2e_pipeline_latency_with_oom_check(
        self, stage_latencies: List[float], parallelism_plan: ParallelismPlan,
    ) -> Optional[float]:
        """ Compute model end-to-end pipeline latency with OOM check. """
        raise NotImplementedError()
