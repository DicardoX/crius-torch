#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import Callable, Iterable

__all__ = [ "get_train_data_iterator", "get_optimizer_and_scheduler", ]

def get_train_data_iterator() -> Iterable:
    """ Get training data iterator. """
    raise NotImplementedError

def get_optimizer_and_scheduler(): 
    """ Get optimizer and learning rate scheduler. """
    raise NotImplementedError
