#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph transformation. """

import os
from typing import (List, Callable, Any, Optional, Union,Dict)
from collections import defaultdict
import re
import logging

import torch
from torch import nn
import torch._dynamo as dynamo
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch._subclasses.fake_tensor import FakeTensorMode

from crius.compile.graph.graph import (IRNode, IRGraph)
from crius.model_parallel.utils import ParallelismPlan
from crius.model.model_config import model_config

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
    'attn_resid_dropout': 'torch.nn.Dropout',
    'mlp_dropout': 'torch.nn.Dropout',
}

call_function_core_op_category = {
    'add': 'torch.add ',
    'arange': 'torch.arange',
    'aten.sym_size.int': 'torch.Tensor.size ',
    'eq': 'torch.eq ',
    'getitem': 'torch.Tensor.__getitem__',
    'matmul': 'torch.matmul ',
    'mul': 'torch.mul ',
    'ones': 'torch.ones',
    'tril': 'torch.tril',
    'cat': 'torch.cat',                            # 沿指定维度拼接张量
    'neg': 'torch.neg',                            # 逐元素取负
    'addmm': 'torch.addmm',
    'pow': 'torch.pow',
    'tanh': 'torch.tanh',

    '_enter_autocast': 'torch.cuda.amp.autocast',  # 进入自动混合精度上下文
    '_exit_autocast': 'torch.cuda.amp.autocast',   # 退出混合精度上下文
    '_set_grad_enabled': 'torch.set_grad_enabled', # 控制是否启动梯度计算
    'rsqrt': 'torch.rsqrt',                        # 逐元素计算1/sqrt(x)
    'scaled_dot_product_attention': 'F.scaled_dot_product_attention()'

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


    match = re.search(r'modules__\d+___(\w+)', target)
    if match:
        return match.group(1)
    
    # 匹配顶层模块，如lm_head
    if target.startswith("L__self___"):
        return target[len("L__self___"):] 
    

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
    
    if target in ('aten._assert_scalar.default', 'aten.sym_constrain_range_for_size.default'):
        return target

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


def replace_where_with_static(graph_module: torch.fx.GraphModule, *input_args, top_k: int):
    """
    Replace all `torch.where(condition)` calls with a fake static version
    that returns two tensors of shape (top_k,).
    """
    graph = graph_module.graph

    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.where:
            # Check if it's the two-arg form: torch.where(condition)
            if len(node.args) == 1 and len(node.kwargs) == 0:
                condition = node.args[0]
                with graph.inserting_after(node):
                    # Create a fake index tensor of shape (top_k,)
                    # Use torch.arange to generate a static-sized tensor
                    fake_idx = graph.create_node(
                        'call_function',
                        torch.arange,
                        args=(top_k,),
                        kwargs={'dtype': torch.int64, 'device': "cuda"}  
                    )
                    # Duplicate for both outputs: (indices, top_x)
                    # You can also use graph.create_node('copy', fake_idx) if needed
                    user_nodes = list(node.users.keys())
                    node.replace_all_uses_with((fake_idx, fake_idx))
                    graph.erase_node(node)
    graph_module.recompile()
    return graph_module


def transform_to_ir_graph(
    model: nn.Module,
    *input_args,
) -> Any:
    """ Transform model into IR graph. """
    # Trace with Python-level JIT compiler
    fx_module = dynamo.export(model)(*input_args)[0]

    # For moe module. get top_k config and replace torch.where 
    if model_config.if_moe == True:
        top_k = model_config.top_k
        logging.debug(f"Moe Model, change torch.where output from top_k {top_k} info ...")
        fx_module = replace_where_with_static(fx_module, input_args, top_k=top_k)

    fx_graph = fx_module.graph
    # Construct IR graph
    graph = IRGraph(fx_graph.owning_module)

    # ShapeProp必须放在graph_copy前面
    from torch.fx.passes.shape_prop import ShapeProp
    try:
        ShapeProp(graph.owning_module).propagate(*input_args)
    except Exception as e:
        raise RuntimeError(f"ShapeProp failed even with patch: {e}") from e

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
    # from torch.fx.passes.shape_prop import ShapeProp
    
    # ShapeProp(graph.owning_module).propagate(*input_args)
    graph = graph.owning_module.graph
    for node in graph.nodes:
        # Input shapes
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node) and hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                tensor_meta = arg.meta['tensor_meta']
                if isinstance(tensor_meta, tuple) and  all(isinstance(x, TensorMetadata) for x in tensor_meta):
                    for meta in tensor_meta:
                        input_shapes.append(meta.shape)
                else:
                    input_shapes.append(tensor_meta.shape)

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

        # dtype
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            if hasattr(tensor_meta, 'dtype'):
                dtype = tensor_meta.dtype
            elif isinstance(tensor_meta, torch.Tensor):
                dtype = tensor_meta.dtype
            else:
                dtype = None
        else:
            dtype = None
        setattr(node, 'dtype', dtype)

    # 再处理 to 算子，设置 pre_dtype
        if node.op in ['call_function', 'call_method'] and node.target in ['to', torch.Tensor.to]:
            source_node = find_source_tensor_node(node)
            pre_dtype = getattr(source_node, 'dtype', None) if source_node else None
            setattr(node, 'pre_dtype', pre_dtype)
        else:
            setattr(node, 'pre_dtype', None)
    
    return graph


def find_source_tensor_node(to_node: fx.Node):
    """
    找到 to 算子的“源张量”输入节点（即真正要被转换的那个张量）。
    """
    TO_ARG_KEYS = {'device', 'dtype', 'non_blocking', 'copy', 'memory_format'}

    # 1. 检查 args
    for arg in to_node.args:
        if not isinstance(arg, fx.Node):
            continue
        # 判断是否是真正的张量（排除 device/dtype 等 scalar）
        if is_tensor_node(arg):
            return arg

    # 2. 检查 kwargs
    for key, value in to_node.kwargs.items():
        if key in TO_ARG_KEYS:
            continue
        if isinstance(value, fx.Node) and is_tensor_node(value):
            return value

    return None


def is_tensor_node(node: fx.Node) -> bool:
    """
    判断一个 Node 是否是张量（通过 shape 判断，排除 scalar）
    """
    if not hasattr(node, 'meta') or 'tensor_meta' not in node.meta:
        return False
    tensor_meta = node.meta['tensor_meta']
    # 如果有 shape 且不是标量，则认为是张量
    if hasattr(tensor_meta, 'shape'):
        return tensor_meta.shape != torch.Size([])
    return True


def infer_node_layer_types(graph: IRGraph) -> IRGraph:
    """ Infer node layer types (e.g., gemm) in the IR graph. 

    NOTE(chunyu): The `node.target` should be strictly named, maybe add pre sanity check at the beginning.
    
    Args:
        graph: IR graph to infer layer types.
    """
    for node in graph.nodes:
        target = str(node.target)
        if hasattr(node, 'layer_type'):
            continue
        
        # GEMM
        if node.op == 'call_module':
            if 'fc' in target:
                setattr(node, 'layer_type', 'linear')
            elif 'proj' in target:
                if 'experts' in target:   #for expert module， eg: L__self___model_layers_0_mlp_experts_0_gate_proj
                    setattr(node, 'layer_type', 'expert_linear')
                else:
                    setattr(node, 'layer_type', 'linear')
            elif 'qkv' in target:
                setattr(node, 'layer_type', 'linear')
            elif 'lm' in target:
                setattr(node, 'layer_type', 'linear')
            elif 'ln' in target:
                setattr(node, 'layer_type', 'layernorm')
            elif 'act' in target:
                if 'experts' in target:   #for expert module， eg: L__self___model_layers_0_mlp_experts_0_act_fn
                    setattr(node, 'layer_type', 'expert_gelu')
                else:
                    setattr(node, 'layer_type', 'gelu')
            elif 'drop'  in target:
                setattr(node, 'layer_type', 'dropout')
            elif 'wte' in target:
                setattr(node, 'layer_type', 'embedding')
            elif 'wpe' in target:
                setattr(node, 'layer_type', 'embedding')
            elif 'embed' in target:
                setattr(node, 'layer_type', 'embedding')
            elif 'gate' in target:
                setattr(node, 'layer_type', 'softmax')
            else:
                print(f"unknown call module: {target}")
                setattr(node, 'layer_type', 'unknown')
        elif node.op =="call_function":
            if 'matmul' in target:
                setattr(node, 'layer_type', 'matmul')
            elif 'mul' in target:
                setattr(node, 'layer_type', 'mul')
            elif 'addmm' in target:
                setattr(node, 'layer_type', 'addmm')
            elif 'add' in target:
                setattr(node, 'layer_type', 'add')
            elif 'scaled_dot_product_attention' in target:
                setattr(node, 'layer_type', 'sdpa')
            elif 'pow' in target:
                setattr(node, 'layer_type', 'pow')
            # elif 'eq' in target:
            #     setattr(node, 'layer_type', 'eq')
            elif 'topk' in target:
                setattr(node, 'layer_type', 'topk')
            elif 'tril' in target:
                setattr(node, 'layer_type', 'tril')
            elif 'softmax' in target:
                setattr(node, 'layer_type', 'softmax')
            elif 'rsqrt' in target:
                setattr(node, 'layer_type', 'rsqrt')
            else:
                print(f"unknown call function: {target}")
                setattr(node, 'layer_type', 'unknown')
        
        elif node.op == "call_method":
            if 'softmax' in target:
                setattr(node, 'layer_type', 'softmax')
            elif 'masked_fill' in target:
                setattr(node, 'layer_type', 'masked_fill')
            elif 'contiguous' in target:
                setattr(node, 'layer_type', 'contiguous')
            elif 'to' in target:
                setattr(node, 'layer_type', 'to')
            elif 'pow' in target:
                setattr(node, 'layer_type', 'pow')
            else:
                print(f"unknown call method: {target}")
                setattr(node, 'layer_type', 'unknown')
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

