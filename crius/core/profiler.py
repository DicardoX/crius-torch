#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Main entrypoint of Crius Torch-like profiler. 

This module is independent to the underlying training backend (e.g., Megatron).
"""

from typing import (Any, Tuple, Optional, List, Callable, Dict)
import numpy as np
import pickle

import torch
from torch import nn

from crius.core.utils import estimate_model_memory_access_gb
from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (compile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode
# from crius.offline.offline_profiler import CriusOfflineDataManager
from crius.offline.offline_recorder import OfflineDataRecorder
from crius.global_envs import logger, global_configs


class CriusDistProfiler:
    """ Crius Torch-like disaggregated profiler. 
    
    Roadmap of this module: (1) Operator disaggregation, and (2) Device-free model profiling.
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
        parallelism_plan: ParallelismPlan,
        only_forward: bool = False,
        base_input_args: Tuple[Any] = None,
        base_input_kwargs: Dict[str, Any] = None,
        graph: IRGraph = None,
        bw_graph: IRGraph = None,
        partition_pipeline_stages_func: Callable = None,
        check_oom: bool = True,
    ) -> Optional[float]:
        """ Profile the model. 
        
        Args:
            model: Torch-like model to be profiled.
            parallelism_plan: Parallelism plan of the model.
            only_forward: If True, only profile the forward pass. Default is False.
            base_input_args: (Optional) Input arguments of model `.forward()` for IR graph transformation.
            base_input_kwargs: (Optional) Input keyword arguments of model `.forward()` for IR graph transformation.
            graph: (Optional) Pre-generated forward IR graph.
            bw_graph: (Optional) Pre-generated backward IR graph.
            check_oom: (Optional): If true, check out-of-memory after stage partition.
        
        Returns:
            e2e_latency: End-to-end pipeline latency of the model in milliseconds, or None
        """

        # Step 0. (Optional) Compile the model into IR graph
        if graph is None:
            assert base_input_args is not None and base_input_kwargs is not None, \
                f"When no pre-generated IR graphs, must provide base inputs for compilation in profiler."
            compile_configs = CompileConfigs(
                base_input_args=base_input_args,
                base_input_kwargs=base_input_kwargs,
                parallelism_plan=parallelism_plan,
            )
            graph, bw_graph = compile(model, compile_configs)
        
        if partition_pipeline_stages_func is not None:
            partition_pipeline_stages_func(graph, parallelism_plan, bw_graph)

        # Step 0. (Optional) Check OOM
        if check_oom:
            (max_stage_weight_mem, max_stage_act_mem, 
            max_stage_optim_mem, max_stage_grad_mem) = estimate_model_memory_access_gb(parallelism_plan, graph, bw_graph)
            if np.sum(
                max_stage_weight_mem + max_stage_act_mem + max_stage_optim_mem + max_stage_grad_mem
            ) > global_configs.per_device_memory_capacity_in_gb:
                return -1.0    # OOM occurs

        # Step 1. Query and record per-operator latencies
        graph, bw_graph = self._query_operator_latencies_ms(graph, parallelism_plan, bw_graph)
        logger.debug("\n\n============> Forward graph")
        for node in graph.nodes:
            prefix_str_, suffix_str_ = "", ""
            if node.layer_type in (
                "gemm", "bmm", "baddbmm", 
                "all_reduce", "all_gather", "all_to_all", "reduce_scatter", 
            ):
                prefix_str_ = "\033[1;37;42m------> [HEAVY-OP] "
                suffix_str_ = "\033[0m"
            latency = getattr(node, "latency", None)
            logger.debug(f"{prefix_str_}Node target: {node.target} | " + 
                         f"OP type: {node.op} | Layer type: {node.layer_type} | Stage index: {node.stage_index} |" + 
                         f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                         f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")
        if bw_graph is not None:
            logger.debug("\n\n============> Backward graph")
            prefix_str_, suffix_str_ = "", ""
            for node in bw_graph.nodes:
                latency = getattr(node, "latency", None)
                logger.debug(f"{prefix_str_}Node target: {node.target} | " + 
                             f"OP type: {node.op} | Layer type: {node.layer_type} | Stage index: {node.stage_index} |" + 
                             f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                             f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")

        # Step 2. Compute per-stage latency
        stage_latencies, grad_sync_latencies = self._compute_stage_latencies(graph, parallelism_plan, bw_graph)  

        # Step 3. Model end-to-end pipeline latency
        e2e_latency = self._model_e2e_pipeline_latency_with_oom_check(
            stage_latencies, grad_sync_latencies, parallelism_plan, only_forward,
        )
        return e2e_latency

    def _query_operator_latencies_ms(self, graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: IRGraph = None):
        """ Query operator latencies(ms) from offline profiled data. """

        def __query_one_graph(graph_): 
            for node in graph_.nodes:
                if (
                    str(node.op) in ("call_module", "call_function", "call_method") and
                    "unknown" not in node.layer_type
                ):
                    node_latency = 0.0
                    if self._profiled_data is not None:
                        # Query the latency from the profiled data
                        input_shapes, output_shapes = node.input_shapes, node.output_shapes
                        batch_size, seq_len, left_op_hidden_dim, right_op_hidden_dim, layout = None, None, None, None, None
                        comm_group_size, comm_data_size = None, None
                        assert parallelism_plan.num_attn_heads % parallelism_plan.tensor_parallel_degree == 0, \
                            f"The number of attention heads should be divisible by TP size."
                        num_attn_heads_per_rank = parallelism_plan.num_attn_heads // parallelism_plan.tensor_parallel_degree

                        if node.layer_type in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter", ):
                            if node.node_parallelism is None:
                                continue    # This node is a not-used placeholder (not defined in comm op insertion 
                                            # rules) in model forward definition.

                            comm_group_size = self._get_comm_group_size_one_node(node, parallelism_plan)
                            comm_data_size = np.prod(list(input_shapes))
                        
                        elif node.layer_type in ("gemm", "bmm", "baddbmm", ):
                            assert getattr(node, "layout", None) is not None
                            layout = [l.split(":") for l in node.layout.split(",")]
                            batch_size_indices = self._get_one_dim_index("b", layout)
                            seq_len_indices = self._get_one_dim_index("s", layout)
                            left_op_hidden_dim_indices = self._get_one_dim_index("h1", layout)
                            right_op_hidden_dim_indices = self._get_one_dim_index("h2", layout)

                            if "*" not in node.layout:
                                batch_size = input_shapes[batch_size_indices[0]][batch_size_indices[1]] \
                                    if batch_size_indices is not None else None
                                seq_len = input_shapes[seq_len_indices[0]][seq_len_indices[1]] \
                                    if seq_len_indices is not None else None
                                left_op_hidden_dim = input_shapes[left_op_hidden_dim_indices[0]][left_op_hidden_dim_indices[1]] \
                                    if left_op_hidden_dim_indices is not None else None
                                right_op_hidden_dim = input_shapes[right_op_hidden_dim_indices[0]][right_op_hidden_dim_indices[1]] \
                                    if right_op_hidden_dim_indices is not None else None
                            else:
                                # Hardcoded when has merged dimensions (e.g., "h1:s*b,s*b:h2")
                                if node.layout == "h1:s*b,s*b:h2":
                                    assert len(input_shapes[0]) == len(input_shapes[1]) == 2
                                    batch_size = 1
                                    seq_len = input_shapes[0][1]    # Accumulated
                                    left_op_hidden_dim = input_shapes[0][0]
                                    right_op_hidden_dim = input_shapes[1][1]
                                else:
                                    raise ValueError(f"Unsupported hardcoded node layout: {node.layout}")
                        
                        else:
                            logger.warning(f"Ignored node type {node.layer_type} in latency query.")
                        
                        query_key = OfflineDataRecorder.get_query_key(
                            node.layer_type, batch_size, seq_len, left_op_hidden_dim, right_op_hidden_dim, 
                            node.layout, num_attn_heads_per_rank, global_configs.num_nodes, comm_group_size, 
                            comm_data_size, attn_head_split=False,
                        )

                        if query_key in self._profiled_data:
                            node_latency = self._profiled_data[query_key]["latency_ms"]
                        elif query_key is not None:
                            logger.warning(f"Query key {query_key} not found in profiled data, using default latency {node_latency}ms.")
                    
                    setattr(node, "latency", node_latency)
                else:
                    # Non dominant nodes
                    setattr(node, "latency", None)
        
        __query_one_graph(graph)
        if bw_graph is not None:
            __query_one_graph(bw_graph)

        checkout_node_latencies = True     # For debug
        if checkout_node_latencies:
            accumulated_latency = 0.0
            for node in graph.nodes:
                if node.layer_type in ("gemm", "bmm", "baddbmm", "all_reduce"):
                    accumulated_latency += node.latency
                    print(node.layer_type, node.input_shapes, node.layout, node.latency, accumulated_latency)

            print("\n\n\n\n\n")
            
            accumulated_latency = 0.0
            for node in bw_graph.nodes:
                if node.layer_type in ("gemm", "bmm", "baddbmm", "all_reduce"):
                    accumulated_latency += node.latency
                    print(node.layer_type, node.input_shapes, node.layout, node.latency, accumulated_latency)
            exit(0)

        return graph, bw_graph
    
    def _get_one_dim_index(self, dim_str: str, layout: List[List[str]]) -> Tuple[int]:
        """ Get the first 2D indices from the layout. 
        
        Example: (input) 'b' -> [['b', 's', 'h1'], ['h1', 'h2']] -> (return) [0, 0]
        """
        for i, sub_layout in enumerate(layout):
            for j, dim in enumerate(sub_layout):
                if dim == dim_str:
                    return (i, j)
        return None
    
    def _get_comm_group_size_one_node(self, node: IRNode, parallelism_plan: ParallelismPlan) -> int:
        """ Get the communication group size for one node. """

        if node.node_parallelism == "tp":
            return parallelism_plan.tensor_parallel_degree
        elif node.node_parallelism == "dp":
            return parallelism_plan.data_parallel_degree
        elif node.node_parallelism == "pp":
            return 2    # P2P communication
        else:
            raise NotImplementedError("Unsupported node parallelism plan...")

    def _compute_stage_latencies(
        self, graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: Optional[IRGraph],
    ) -> Tuple[List[List[float]], List[float]]:
        """ Compute per-stage latencies. """

        stage_latencies = [[0.0, 0.0] for _ in range(parallelism_plan.pipeline_parallel_degree)]
        grad_sync_latencies = [0.0 for _ in range(parallelism_plan.pipeline_parallel_degree)]
        for i, node in enumerate(list(graph.nodes)):
            # Accumulate latency
            assert node.stage_index is not None, "Forward node stage index is not set."
            if node.latency is not None and node.stage_index is not None:
                stage_latencies[node.stage_index][0] += node.latency
        
        if global_configs.only_forward:
            return stage_latencies
        
        if not global_configs.enable_backward_graph_generation:
            # For backward pass, we assume the latency is 2x of the forward pass.
            logger.warning("No backward graph generation, use '2x forward stage latency' to estimate backward stage latency.")
            for i in range(parallelism_plan.pipeline_parallel_degree):
                stage_latencies[i][1] = stage_latencies[i][0] * 2.0
        else:
            assert bw_graph is not None
            for node in bw_graph.nodes:
                assert node.stage_index is not None, "Backward node stage index is not set."
                if node.latency is None or node.stage_index is None:
                    continue

                if (node.node_pass == "backward" and node.node_parallelism == "dp" and node.layer_type == "all_reduce"):
                    # The gradient sync of DP is conduncted at last per-batch instead of per-microbatch. We should seperate this
                    # rather than accounting it in the latency of backward stages.
                    grad_sync_latencies[node.stage_index] += node.latency
                else:
                    # Normal backward nodes
                    stage_latencies[node.stage_index][1] += node.latency

        print(stage_latencies)
        exit(0)

        return stage_latencies, grad_sync_latencies

    def _model_e2e_pipeline_latency_with_oom_check(
        self, stage_latencies: List[List[float]], grad_sync_latencies: List[float], 
        parallelism_plan: ParallelismPlan, only_forward: bool = False,
    ) -> Optional[float]:
        """ Compute model end-to-end pipeline latency with OOM check. """
        
        if parallelism_plan.pipeline_type == "1f1b":
            return self._model_e2e_pipeline_latency_with_oom_check_1f1b_schedule(
                stage_latencies, grad_sync_latencies, parallelism_plan, only_forward,
            )
        else:
            raise ValueError(f"Pipeline type {parallelism_plan.pipeline_type} is not supported yet.")
    
    def _model_e2e_pipeline_latency_with_oom_check_1f1b_schedule(
        self, stage_latencies: List[List[float]], grad_sync_latencies: List[float], 
        parallelism_plan: ParallelismPlan, only_forward: bool = False,
    ) -> Optional[float]:
        """ 1F1B pipeline schedule. 

        An example with 2 stages (2 fw + 2 bw) and 2 microbatches:

                   Warmup |         Steady          | Drain
        clock num    0    |   1     2     3     4   |   5
        ---------  -----  | ----- ----- ----- ----- | -----
         stage 0   (0,0)  | (0,1)       (0,-1)       | (0,-2)
         stage 1          | (1,0) (1,-1) (1,1) (1,-2) |

        where `(i,j)` denotes stage `i` executes forward micro-batch `j` when `j >= 0`,
        and denotes stage `i` executes backward micro-batch `-j - 1` when `j < 0`.
        """

        print(stage_latencies)
        print(grad_sync_latencies)
        exit(0)
        
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
        # NOTE: Approximation: grad sync overhead remains nearly similar across stages
        t_drain = np.sum([stage_latencies[i][1] for i in range(num_stages - 1)]) + np.max(grad_sync_latencies)

        return (t_warmup + t_steady + t_drain)
