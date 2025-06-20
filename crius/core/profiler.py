#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Main entrypoint of Crius Torch-like profiler. 

This module is independent to the underlying training backend (e.g., Megatron).
"""

from typing import (Any, Tuple, Optional)

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
    ) -> None:
        """ Profile the model. 
        
        Args:
            model: Torch-like model to be profiled.
            base_input_args: Input arguments of model `.forward()` for IR graph transformation.
            parallelism_plan: Parallelism plan of the model.
        """

        # Step 1. Compile the model into IR graph
        compile_configs = CompileConfigs(
            base_input_args=base_input_args,
            parallelism_plan=parallelism_plan,
        )
        graph = compile(model, compile_configs)

        # Step 2. Query per-operator latency
        for node in graph.nodes:
            if node.op == "call_module":
                setattr(node, "latency", self._query_operator_latency_ms(node))
            else:
                setattr(node, "latency", None)

        for node in graph.nodes:
            print(f"\n=====> Node target: {node.target}")
            print(f"Node: {node.name} | Node class: {node.__class__} | OP type: {node.op} | " + 
                  f"Next: {node.next} | Input shapes: {node.input_shapes} | " + 
                  f"Output shapes: {node.output_shapes} | Latency: {node.latency}")

        # Step 3. Computing per-stage latency
        stage_latencies = [0.0] * parallelism_plan.pipeline_parallel_degree
        for node in graph.nodes:
            raise NotImplementedError()

    def _query_operator_latency_ms(self, node: IRNode):
        """ Query operator latency(ms) from offline profiled data. """
        
        if self._profiled_data is None:
            return 1.0   # For debug

        NotImplementedError()
