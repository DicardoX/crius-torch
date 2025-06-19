#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Implementation of IR objects for automatic transformation of computational graph. """

from typing import (
    Union, Callable, Any, Tuple, List, Optional, Dict, Type)
import copy

import torch
import torch.fx as fx
from torch.fx.node import (Target, Argument, map_arg)
from torch.fx import GraphModule

__all__ = [
    "IRNode",
    "IRGraph",
]


class IRNode(fx.Node):
    """ IR node object to support automatic graph transformation.
    
    Args:
        graph: The `Graph` to which this `Node` should belong.
        name: The name of this `Node`.
        op: The opcode for this `Node`. Can be one of `placeholder`, `call_method`, `call_module`, 
            `call_function`, `get_attr`, `output`.
        target: The target this op should call (e.g., function object if `op` is `call_function`).
        args: The args to be passed to `target`.
        kwargs: The kwargs to be passed to `target`.
        return_type: The python type expression representing the type of the output of this node. 
                     This field can be used for annotation of values in the generated code or for 
                     other types of analyses.
    """

    def __init__(
        self,
        graph: Union["IRGraph", fx.Graph],
        name: str,
        op: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        return_type: Optional[Any] = None,
    ) -> None:
        """ Initialize a IR node object. """

        super().__init__(graph, name, op, target, args, kwargs, return_type)


class IRGraph(fx.Graph):
    """ IR graph object to support automatic graph transformation. 
    
    Args:
        owning_module: Torch module which this graph belongs to.
    """

    def __init__(
        self,
        owning_module: Optional[GraphModule] = None,
    ) -> None:
        """ Initialize a IR graph object. """
        super().__init__(owning_module)
        self._root: IRNode = IRNode(self, "", "root", "", (), {})
        self._insert = self._root.prepend   # Iterating direction is `_prev` in `g.nodes`

    def node_copy(
        self, node: fx.Node, arg_transform: Callable[[IRNode], Argument] = ...
    ) -> IRNode:
        """ Construct an `IRNode` from the given `fx.Node`. """
        if isinstance(node, IRNode):
            args = map_arg(node.args, arg_transform)
            kwargs = map_arg(node.kwargs, arg_transform)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            result_node = self.create_node(
                node.op, node.target, args, kwargs, node.name, node.type,
            )
            result_node.meta = copy.copy(node.meta)
            return result_node
        else:
            assert isinstance(node, fx.Node)
            return super().node_copy(node, arg_transform)

    def create_node(
        self,
        op: str,
        target: Target,
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> IRNode:
        assert op in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        )
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        
        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = IRNode(self, name, op, target, args, kwargs, type_expr)
        self._graph_namespace.associate_name_with_obj(name, n)
        self._insert(n)
        self._len += 1
        return n
