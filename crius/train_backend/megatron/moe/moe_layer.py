#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's MoE layer modules. """

from typing import Optional, Union

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [ "MoELayer", ]


class MoELayer(MegatronModule):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        ep_size = 1     # TODO(chunyu): Currently we don't support expert parallelism.
        
        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size

        self.router = None  # TODO(chunyu): Implement router initialization here.

        assert config.moe_token_dispather_type == "allgather", \
            "Only All-Gather dispatcher is supported currently."
        self.dispatcher = None  # TODO(chunyu): Implement All-Gather dispatcher here.

        raise NotImplementedError("MoELayer is not implemented yet.")
