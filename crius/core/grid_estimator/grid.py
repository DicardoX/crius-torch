#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue


class Grid:
    """ Grid object for efficient parallelism optimization. """

    def __init__(self, num_nodes: int, num_devices_per_node: int, pipeline_degree: int):
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.pipeline_degree = pipeline_degree

        self.pareto_optimal_plans = None
