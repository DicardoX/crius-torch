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
import re
from torch import nn

from crius.model_parallel.utils import ParallelismPlan
from crius.compile import (compile, CompileConfigs)
from crius.compile.graph import IRGraph, IRNode
from crius.offline.offline_recorder import OfflineDataRecorder
from crius.global_envs import logger, global_configs

import torch.distributed as dist
save_node_info = True

class CriusDistProfiler:
    """ Crius Torch-like disaggregated profiler. 
    
    This module provides (1) operator disaggregation and (2) Device-free model profiling.
    """

    def __init__(
        self, 
        profiled_cmpt_data_path: Optional[str] = None, 
        profiled_comm_data_path: Optional[str] = None
    ) -> None:
        self._profiled_cmpt_data = self._load_profiled_data(profiled_cmpt_data_path)
        self._profiled_comm_data = self._load_profiled_data(profiled_comm_data_path)

        self.MAX_COMM_SIZE = 1 << 28
        self.LARGE_COMM_SCALE_FACTOR = 1.1
        
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
        base_input_kwargs: Dict[str, Any],
        parallelism_plan: ParallelismPlan,
        only_forward: bool = False,
    ) -> Optional[float]:
        """ Profile the model. 
        
        Args:
            model: Torch-like model to be profiled.
            base_input_args: Input arguments of model `.forward()` for IR graph transformation.
            base_input_kwargs: Input keyword arguments of model `.forward()` for IR graph transformation.
            parallelism_plan: Parallelism plan of the model.
            only_forward: If True, only profile the forward pass. Default is False.
        
        Returns:
            e2e_latency: End-to-end pipeline latency of the model in milliseconds, or None
        """

        # Step 1. Compile the model into IR graph
        compile_configs = CompileConfigs(
            base_input_args=base_input_args,
            base_input_kwargs=base_input_kwargs,
            parallelism_plan=parallelism_plan,
        )
        graph, bw_graph = compile(model, compile_configs)

        # Step 2. Query and record per-operator latencies
        graph, bw_graph = self._query_operator_latencies_ms(graph, parallelism_plan, bw_graph)
        
        # print(f'graph: {graph}')
        # print(f'bw_graph: {bw_graph}')

        # logger.debug("\n\n============> Forward graph")

        for node in graph.nodes:
            prefix_str_, suffix_str_ = "", ""
            if node.layer_type in (
                "gemm", "reduce_sum", "bmm", "baddbmm", 
                "all_reduce", "all_gather", "all_to_all", "reduce_scatter", 
            ):
                prefix_str_ = "\033[1;37;42m------> [HEAVY-OP] "
                suffix_str_ = "\033[0m"
            latency = getattr(node, "latency", None)
            if dist.get_rank() == 0 and save_node_info == True:
                with open('/app/scripts/c_graph_dp2.txt', 'a', encoding = 'utf-8') as f:
                    f.write(f"{prefix_str_}Node target: {node.target} | " + 
                            f"OP type: {node.op} | Layer type: {node.layer_type} | " + 
                            f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                            f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")

                # logger.debug(f"{prefix_str_}Node target: {node.target} | " + 
                #             f"OP type: {node.op} | Layer type: {node.layer_type} | " + 
                #             f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                #             f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")
        if bw_graph is not None:
            logger.debug("\n\n============> Backward graph")
            prefix_str_, suffix_str_ = "", ""
            for node in bw_graph.nodes:
                latency = getattr(node, "latency", None)
                if dist.get_rank() == 0 and save_node_info == True:
                    with open('/app/scripts/c_graph_dp2.txt', 'a', encoding = 'utf-8') as f:
                        f.write(f"{prefix_str_}Node target: {node.target} | " + 
                                f"OP type: {node.op} | Layer type: {node.layer_type} | " + 
                                f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                                f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")

                    # logger.debug(f"{prefix_str_}Node target: {node.target} | " + 
                    #              f"OP type: {node.op} | Layer type: {node.layer_type} | " + 
                    #              f"Input shapes: {node.input_shapes} | Output shapes: {node.output_shapes} | " + 
                    #              f"Latency: {latency} | Layout: {node.layout}{suffix_str_}\n")

        # Step 3. Compute per-stage latency
        stage_latencies = self._compute_stage_latencies(graph, parallelism_plan, bw_graph)  

        # Step 4. Model end-to-end pipeline latency
        e2e_latency = self._model_e2e_pipeline_latency_with_oom_check(
            stage_latencies, parallelism_plan, only_forward,
        )
        return e2e_latency

    def _query_operator_latencies_ms(self, graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: IRGraph = None):
        """ Query operator latencies(ms) from offline profiled data. """

        def _get_comm_time_table() -> dict:
            """ Match target patterns from the loaded content. """
            pattern = r'^(all_reduce|all_gather|all_to_all|reduce_scatter)_n(\d+)_gs(\d+)_d(\d+)$'
            comm_time_table = {}

            for key, vdict in self._profiled_comm_data.items():
                match = re.match(pattern, key)
                if match:
                    op_type = match.group(1)       
                    n  = int(match.group(2))         
                    gs = int(match.group(3))        
                    d  = int(match.group(4))         
                else:
                    raise ValueError(f"Unable to parse key: {key}")  
                    
                _key = f'{op_type}_n{n}_gs{gs}'
                if _key not in comm_time_table:
                    comm_time_table[_key] = []
                else:
                    comm_time_table[_key].append([d, vdict['latency_ms']])
                    
            # print(comm_time_table)                 

            return comm_time_table

        def _estimate_comm_op_time(
            op_type: str, 
            num_node: int, 
            group_size: int, 
            comm_size: int, 
            # dtype: Any, 
            comm_time_table: dict, 
        ):
            """ Estimate the time of the communication operator based on offline profiling results. """
            # Querying key
            key = f'{op_type}_n{num_node}_gs{group_size}'
            if key not in comm_time_table:
                logger.warning(f"Query key [ {key} ] not found in profiled data, using default latency 0 ms.")
                return 0

            # Estimate comm time
            last_comm_size, last_comm_time = -1, -1
            # dtype = str(dtype)

            # FIXME(junzhou): Assuming that comm_size denotes the number of parameter, and default type is torch.float16 . 
            # if dtype in DTYPE_TO_F32_RATIO_TABLE:
            #     # Resize comm_size based on dtype
            #     comm_size *= DTYPE_TO_F32_RATIO_TABLE[dtype]
            #     dtype = "float32"

            # print(comm_time_table)

            if comm_size <= comm_time_table[key][0][0]:
                return comm_time_table[key][0][1]
            
            for _i, (_comm_size, _comm_time) in enumerate(comm_time_table[key]):
                # assert dtype == "float32", f"Currently only support data type with float32, got {dtype}."
        
                if _comm_size >= comm_size:
                    # Linear interpolate
                    _ratio = (comm_size - last_comm_size) / (_comm_size - last_comm_size)
                    return last_comm_time + (_comm_time - last_comm_time) * _ratio
                last_comm_size, last_comm_time = _comm_size, _comm_time

            # Exceed max profiled comm size
            max_profiled_comm_size = comm_time_table[key][-1][0]
            # Decompose communication size into profiled scope
            max_profiled_cnt = comm_size // max_profiled_comm_size
            rest_comm_time = _estimate_comm_op_time(op_type, num_node, group_size, (comm_size % max_profiled_comm_size,), comm_time_table)
            max_comm_time = comm_time_table[key][-1][1]

            # if True:
            if max_profiled_comm_size < MAX_COMM_SIZE // 2:
                # Only for low bandwidth cases, which linearly scale the comm time
                est_comm_time = max_profiled_cnt * max_comm_time + rest_comm_time
            else:    
                # When comm amount is extremely large, comm time scales a little faster than linearly scaling 
                # probably due to network interference.
                est_comm_time = (max_profiled_cnt * max_comm_time + rest_comm_time) \
                                * pow(LARGE_COMM_SCALE_FACTOR, max_profiled_cnt)
            
            print(f"[WARN] Communication size of '{op_type}' excceeds the maximum profiled size. Linear scaling is applied.")
            print(f"       - Communication size of the operator: {comm_size} | Estimated comm time: {est_comm_time}")
            print(f"       - Max profiled communication size: {max_profiled_comm_size} | Comm time: {max_comm_time}")
            
            return est_comm_time
        
        def __query_one_graph(graph_): 
            for node in graph_.nodes:
                # if dist.get_rank() == 0:
                #     print(node.node_parallelism)
                if (
                    str(node.op) in ("call_module", "call_function", "call_method") and
                    "unknown" not in node.layer_type
                ):
                    node_latency = 0.0
                    if (self._profiled_cmpt_data is not None) and (self._profiled_comm_data is not None):
                        # Query the latency from the profiled data
                        input_shapes, output_shapes = node.input_shapes, node.output_shapes
                        batch_size, seq_len, left_op_hidden_dim, right_op_hidden_dim, layout = None, None, None, None, None
                        num_node, comm_group_size, comm_data_size = None, None, None

                        if node.layer_type in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter", ):
                            if node.node_parallelism is None:
                                continue    # This node is a not-used placeholder (not defined in comm op insertion 
                                            # rules) in model forward definition.

                            num_node = global_configs.num_nodes
                            comm_group_size = self._get_comm_group_size_one_node(node, parallelism_plan)
                            comm_data_size = np.prod(list(input_shapes))
                        
                        elif node.layer_type in ("gemm", "reduce_sum", "bmm", "baddbmm", ):
                            assert getattr(node, "layout", None) is not None
                            layout = [l.split(":") for l in node.layout.split(",")]
                            batch_size_indices          = self._get_one_dim_index("b", layout)
                            seq_len_indices             = self._get_one_dim_index("s", layout)
                            left_op_hidden_dim_indices  = self._get_one_dim_index("h1", layout)
                            right_op_hidden_dim_indices = self._get_one_dim_index("h2", layout)

                            batch_size = input_shapes[batch_size_indices[0]][batch_size_indices[1]] \
                                if batch_size_indices is not None else None
                            seq_len = input_shapes[seq_len_indices[0]][seq_len_indices[1]] \
                                if seq_len_indices is not None else None
                            left_op_hidden_dim = input_shapes[left_op_hidden_dim_indices[0]][left_op_hidden_dim_indices[1]] \
                                if left_op_hidden_dim_indices is not None else None
                            right_op_hidden_dim = input_shapes[right_op_hidden_dim_indices[0]][right_op_hidden_dim_indices[1]] \
                                if right_op_hidden_dim_indices is not None else None
                        
                        else:
                            raise ValueError(f"Unsupported node layer type: {node.layer_type}")
                        
                        query_key = OfflineDataRecorder.get_query_key(
                            node.layer_type, batch_size, seq_len, left_op_hidden_dim, right_op_hidden_dim, node.layout,  
                            comm_group_size, comm_data_size,
                        )
                        if   query_key in self._profiled_cmpt_data:
                            node_latency = self._profiled_cmpt_data[query_key]["latency_ms"]
                        elif query_key in self._profiled_comm_data:
                            node_latency = self._profiled_comm_data[query_key]["latency_ms"]
                        else:
                            """
                            COMM INTERPOLATION
                            """
                            if node.layer_type in ("all_reduce", "all_gather", "all_to_all", "reduce_scatter", ):
                                if node.node_parallelism is None:
                                    continue    # This node is a not-used placeholder (not defined in comm op insertion 
                                                # rules) in model forward definition.

                                comm_time_table = _get_comm_time_table()
                                node_latency = _estimate_comm_op_time(node.layer_type, num_node, comm_group_size, comm_data_size, comm_time_table)
                            else:
                                logger.warning(f"Query key [ {query_key} ] from node {node.target} not found in profiled data, using default latency {node_latency}ms.")
                    
                    setattr(node, "latency", node_latency)
                else:
                    # Non dominant nodes
                    setattr(node, "latency", None)
        
        __query_one_graph(graph)
        if bw_graph is not None:
            __query_one_graph(bw_graph)

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

    def _compute_stage_latencies(self, graph: IRGraph, parallelism_plan: ParallelismPlan, bw_graph: Optional[IRGraph]) -> List[List[float]]:
        """ Compute per-stage latencies. """

        stage_latencies = [[0.0, 0.0] for _ in range(parallelism_plan.pipeline_parallel_degree)]
        for node in graph.nodes:
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
                if node.latency is not None and node.stage_index is not None:
                    stage_latencies[node.stage_index][1] += node.latency

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
