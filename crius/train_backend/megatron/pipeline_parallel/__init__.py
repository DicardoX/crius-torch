#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""流水线并行模块 - 基于 Megatron-LM 官方实现"""

from . import schedules
from . import p2p_communication

__all__ = [
    'schedules',
    'p2p_communication',
]

