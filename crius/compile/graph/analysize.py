import torch
import torch.fx as fx
import re

from collections import defaultdict
from typing import Dict, List, Tuple
from crius.compile.graph.graph import IRGraph, IRNode
from crius.compile.graph.transform import (call_function_core_op_category, 
                                           clean_target_name, extract_op_name,
                                           call_module_core_op_category2)
# from crius.offline.offline_profiler import operator_online_profile


def get_op_layer_mapping(graph: IRGraph) -> Dict[str, int]:
    """
    根据 torch.fx.graph 获取算子与层次结构的映射关系。
    
    Args:
        graph: torch.fx.graph
        
    Returns:
    """
    op_layer_mapping = {}
    layer_patterns = [
        r'h_(\d+)',                      # GPT-2/GPT-Neo 风格
        r'blocks\.(\d+)',                # Vision Transformer 风格
        r'layers\.(\d+)',                # T5/BART 风格
        r'transformer\.layer\.(\d+)'     # BERT 风格
    ]

    def extract_layer_id(targget:str) -> int:
        for pattern in layer_patterns:
            match = re.search(pattern, targget)
            if match:
                return int(match.group(1))
        return -1
    
    layer_id = 0
    old_layer_id = 0
    for node in graph.nodes:
        target = str(node.target)
        layer_id = extract_layer_id(target)
        if layer_id == -1:
            layer_id = old_layer_id
        else:
            old_layer_id = layer_id

        # layer_ops[layer_id].append(node)
        op_layer_mapping[target] = layer_id
    
    return op_layer_mapping

def analyze_model_layers_from_graph(graph: IRGraph) -> Dict[int, List[IRNode]]:
    """
    根据 torch.fx.graph 分析transformer模型的层次结构和每层的算子。
    
    Args:
        graph: torch.fx.graph
        
    Returns:
        key is the module layer, for example: 'layer_0' 
        the value is a list of operator names, current just include call_module for example ['']
    """
    # 存储每个模块路径下的算子
    
    layer_patterns = [
        r'h_(\d+)',                      # GPT-2/GPT-Neo 风格
        r'blocks\.(\d+)',                # Vision Transformer 风格
        r'layers\.(\d+)',                # T5/BART 风格
        r'transformer\.layer\.(\d+)'     # BERT 风格
    ]

    def extract_layer_id(targget:str) -> int:
        for pattern in layer_patterns:
            match = re.search(pattern, targget)
            if match:
                return int(match.group(1))
        return -1
    
    layer_id = 0
    old_layer_id = 0
    layer_target: Dict[int, List[str]] = defaultdict(list)
    
    for node in graph.nodes:
    
        target = str(node.target)
        layer_id = extract_layer_id(target)
        if layer_id == -1:
            layer_id = old_layer_id
        else:
            old_layer_id = layer_id

        # layer_ops[layer_id].append(node)
        layer_target[layer_id].append(node)


    for layer_id, _ in layer_target.items():
        nodes = layer_target[layer_id]
        print(f"Transformer layer {layer_id} contains {len(nodes)} nodes")

        if layer_id == 0:
            analysis_first_layer(nodes)
        elif len(layer_target) > 3 and layer_id == 1:
            analysis_middle_layer(nodes)
        elif layer_id == len(layer_target) - 1:
            analysis_last_layer(nodes)


    return dict(layer_target)


def analysis_first_layer(layer_nodes: List[IRNode]) -> None:
    """
    分析Transformer模型的第一层
    """


def analysis_middle_layer(layer_nodes: List[IRNode]) -> None:
    """
    分析Transformer模型的中间层
    """

    id = 1
    for node in layer_nodes:
        if node.op == 'call_module':
            op_name = extract_op_name(node.target)
            print(f"{id:<5} | {node.op:<20} | {node.name:<20}  |  {node.target:<30} | {call_module_core_op_category2[op_name]} ")
        elif node.op == 'call_method':
            print(f"{id:<5} | {node.op:<20} | {node.name:<20}  |  {node.target} ")
        elif node.op in ("call_function"):
            target = str(node.target)
            op_name = clean_target_name(target)
            print(f"{id:<5} | {node.op:<20} | {node.name:<20}  |  {op_name} ")

            op = call_function_core_op_category[op_name]
            # if op in ("matmul"):
            #     operator_online_profile(op,)
        # else:
        id += 1

def analysis_last_layer(layer_nodes: List[IRNode]) -> None:
    """
    分析Transformer模型最后一层
    """

def is_moe_model(forward_graph: fx.Graph) -> bool:
    """
    判断给定的前向计算图是否是MOE模型
    """

    for node in forward_graph.node:
        # 坚持是否有专家选择机制
        if node.op == 'call_function' and node.target == 'topk':
            return True
        
    return False    