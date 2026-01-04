#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MoE token dispatcher modules. """

from typing import List, Optional, Tuple

from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [ "MoEAllGatherTokenDispatcher", ]


class MoEAllGatherTokenDispatcher:
    """
    AllGather Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
    ) -> None:
        """Initialize the AllGather based token dispatcher.

        Args:
            num_local_experts (int): Number of local experts.
            local_expert_indices (List[int]): Indices of local experts.
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        raise NotImplementedError("MoE All-Gather Token Dispatcher is not implemented yet.")
    