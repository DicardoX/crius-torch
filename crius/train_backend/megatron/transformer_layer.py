#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Compilable implementations of Megatron's transformer layers. """

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer with static submodules for compilation.

    Args:
        config: Transformer configuration in Megatron.
    """

    def __init__(
        self, 
        config: TransformerConfig, 
    ) -> None:
        super().__init__(config=config)

        # [Module 1: Input Layernorm] Optional Layernorm on the input data

        # [Module 2: SelfAttention]

        # [Module 3: BiasDropoutFusion]

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn

        # [Module 5: CrossAttention]

        # [Module 6: BiasDropoutFusion]

        # [Module 7: Pre MLP] Optional Layernorm before MLP

        # [Module 8: MLP block]

        # [Module 9: BiasDropoutFusion]

        raise NotImplementedError("TransformerLayer is not implemented yet.")
