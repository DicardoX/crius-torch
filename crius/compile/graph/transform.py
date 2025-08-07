#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph transformation. """

import os
from typing import (List, Callable, Any, Optional, Union,Dict)
from collections import defaultdict
import re

import torch
from torch import nn
import torch._dynamo as dynamo

from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan

__all__ = [
    "transform_to_ir_graph",
    "propagate_node_shapes",
    "infer_node_layer_types",
    "insert_comm_primitives_into_graph",
    "print_model_graph",
    "analyze_graph_performance",
    "print_operator_summary"
]

module_core_op_category = {
    'attn_qkv': 'attention',
    'attn_proj': 'attention',
    'attn_attn_dropout': 'attention',
    'mlp_c_fc': 'mlp',
    'mlp_act': 'mlp',
    'mlp_c_proj': 'mlp',
    'ln': 'norm',
    'wte': 'embedding',
    'wpe': 'embedding',
    'drop': 'dropout',
}

call_module_core_op_category2 = {
    'attn_qkv': 'torch.nn.Linear',
    'attn_proj': 'torch.nn.Linear',
    'attn_attn_dropout': 'torch.nn.Dropout',
    'mlp_c_fc': 'torch.nn.Linear',
    'mlp_act': 'torch.F.gelu',
    'mlp_c_proj': 'torch.nn.Linear',
    'ln': 'torch.nn.LayerNorm',
    'wte': 'torch.nn.Embedding',
    'wpe': 'torch.nn.Embedding',
    'drop': 'torch.nn.Dropout',
}

call_function_core_op_category = {
    'add': 'torch.add / torch.Tensor.__add__',
    'arange': 'torch.arange',
    'aten.sym_size.int': 'torch.Tensor.size / torch.ops.aten.sym_size',
    'eq': 'torch.eq / torch.Tensor.__eq__',
    'getitem': 'torch.Tensor.__getitem__',
    'matmul': 'torch.matmul / x @ y',
    'mul': 'torch.mul / torch.Tensor.__mul__',
    'ones': 'torch.ones',
    'tril': 'torch.tril',
}


def extract_op_name(target: str):
    """
    提取算子的核心名称，忽略层编号，并统一 LayerNorm 模块为 'ln'
    同时保留 mlp_*, attn_* 等模块的细粒度名称
    """

    # 匹配 Transformer 层内模块：如 h_0_mlp_c_fc -> mlp_c_fc
    h_match = re.search(r'h_\d+_(ln_\d+|ln_f|mlp_.+|attn_.+|drop)', target)
    if h_match:
        inner_name = h_match.group(1)
        # 统一所有 LayerNorm 模块为 'ln'
        if inner_name.startswith('ln_') or inner_name == 'ln_f':
            return 'ln'
        else:
            return inner_name  # 保留 mlp_*, attn_*, drop 等

    # 匹配顶层模块：如 wte、wpe、ln_f、drop
    top_level_match = re.search(r'_(wte|wpe|drop|ln_f)$', target)
    if top_level_match:
        inner_name = top_level_match.group(1)
        if inner_name == 'ln_f':
            return 'ln'
        return inner_name

    return None


def clean_target_name(target: str):
    """
    输入一个 call_function 的 target 字符串（如 '<built-in function getitem>'）
    输出清洗后的函数名（如 'getitem'）
    支持常见函数、aten.sym_size.int 等
    """

    # 匹配 aten.sym_size.int
    if 'sym_size' in target:
        return 'aten.sym_size.int'

    # 匹配 built-in function
    if 'built-in function' in target:
        return target.split('built-in function ')[1].split('>')[0].strip()

    # 匹配 built-in method
    if 'built-in method' in target:
        return target.split('built-in method ')[1].split(' of ')[0].strip()

    # 匹配 torch.function 或 F.xxx
    if 'function' in target:
        func_name = target.split('function ')[1].split(' at ')[0].strip()
        if 'relu' in func_name:
            return 'F.relu'
        elif 'softmax' in func_name:
            return 'F.softmax'
        elif 'layer_norm' in func_name:
            return 'F.layer_norm'
        elif 'dropout' in func_name:
            return 'F.dropout'
        return func_name

    # 匹配 torch.ops.xxx
    if 'torch.ops.' in target:
        return target.split('torch.ops.')[1].split(' at ')[0].strip()

    # 未知函数
    return 'unknown'        


def count_core_function_ops(graph: IRGraph) -> None:
    op_counter = defaultdict(int)
    for node in graph.nodes:
        op_type = node.op
        if op_type in ("call_function"):
            target = str(node.target)
            # print(target)  
            op_name = clean_target_name(target)
            if op_name:
                op_counter[op_name] += 1

    return op_counter            

def count_core_module_ops(graph: IRGraph) -> None:
    op_counter = defaultdict(int)
    for node in graph.nodes:
        op_type = node.op
        if op_type in ("call_module"):
            target = str(node.target)
            op_name = extract_op_name(target)
            if op_name:
                op_counter[op_name] += 1

    return op_counter   

def count_core_method_ops(graph: IRGraph) -> None:
    op_counter = defaultdict(int)
    for node in graph.nodes:
        op_type = node.op
        if op_type == "call_method":
            target = str(node.target)  
            if target:
                op_counter[target] += 1       

    return op_counter            


def analyze_graph_performance(graph: IRGraph) -> Dict[str, int]:
    """ Analyze the performance of the graph by counting operator types.

    Args:
        graph (IRGraph): The IR graph to analyze.

    Returns:
        Dict[str, int]: A dictionary with operator types as keys and their counts as values.
    """
    operator_counts = {}
    for node in graph.nodes:
        op_type = node.op
        if op_type in operator_counts:
            operator_counts[op_type] += 1
        else:
            operator_counts[op_type] = 1
    return operator_counts

def print_operator_summary(operator_counts: Dict[str, int]) -> None:
    """ Print a summary of operator types and their counts.

    Args:
        operator_counts (Dict[str, int]): A dictionary with operator types and their counts.
    """
    print("Operator Type Summary:")
    for op_type, count in operator_counts.items():
        print(f"  {op_type}: {count}")



def transform_to_ir_graph(
    model: nn.Module,
    *input_args,
) -> Any:
    """ Transform model into IR graph. """
    # Trace with Python-level JIT compiler
    fx_module = dynamo.export(model)(*input_args)[0]
    fx_graph = fx_module.graph
    # Construct IR graph
    graph = IRGraph(fx_graph.owning_module)
    output_vals = graph.graph_copy(fx_graph, val_map={}, return_output_node=True)
    output_val, old_output_node = output_vals
    graph.output(output_val, type_expr=getattr(old_output_node, "type", None))   # Insert output node
    return graph


def propagate_node_shapes(graph: IRGraph, *input_args) -> IRGraph:
    """ Propagate per-node input/output shapes in the IR graph. 
    
    Args:
        graph: IR graph to propagate shapes.
        input_args: Input arguments of the model `.forward()` for shape propagation.
    """
    from torch.fx.passes.shape_prop import ShapeProp
    
    ShapeProp(graph.owning_module).propagate(*input_args)
    graph = graph.owning_module.graph
    for node in graph.nodes:
        # Input shapes
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node) and hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                input_shapes.append(arg.meta['tensor_meta'].shape)
        setattr(node, 'input_shapes', input_shapes)
        
        # Output shapes
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            if not isinstance(tensor_meta, list):
                tensor_meta = [tensor_meta]
            output_shapes = [_meta.shape for _meta in tensor_meta if hasattr(_meta, 'shape')]
        else:
            output_shapes = []
        setattr(node, 'output_shapes', output_shapes)
    
    return graph


def infer_node_layer_types(graph: IRGraph) -> IRGraph:
    """ Infer node layer types (e.g., gemm) in the IR graph. 

    NOTE(chunyu): The `node.target` should be strictly named, maybe add pre sanity check at the beginning.
    
    Args:
        graph: IR graph to infer layer types.
    """
    for node in graph.nodes:
        if hasattr(node, 'layer_type'):
            continue
        
        # GEMM
        if node.op == 'call_module':
            if (
                # ------------- String matching rules -------------
                'fc' in node.target
            ):
                setattr(node, 'layer_type', 'gemm')
            else:
                setattr(node, 'layer_type', 'unknown')

    return graph


def insert_comm_primitives_into_graph(
    graph: IRGraph,
    parallelism_plan: ParallelismPlan,
 ) -> IRGraph:
    """ Insert IR nodes of communication primitives in IR graph, based on the parallelism plan. 
    
    Args:
        graph: IR graph to be inserted with communication primitives.
        parallelism_plan: How the to-be-compiled model would be parallelized.
    """
    # TODO(chunyu): Implement this function.
    return graph

def print_model_graph(graph: IRGraph):
    print(f"Graph Length: {len(graph)}")
    print("Graph Nodes:")
    for node in graph.nodes:
        print(f"  {node}")
        print(f"    Operator Name: {node.name}")
        print(f"    Operator Op: {node.op}")
        print(f"    Operator Target: {node.target}")
        print("-" * 40)

