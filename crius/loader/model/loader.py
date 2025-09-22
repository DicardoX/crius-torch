#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" General model loader. """

from typing import (Optional, Callable, Dict, Any)

import torch
from torch import nn

__all__ = [
    "BaseModelLoader",
    "ModelLoader",
]


class BaseModelLoader:
    """ Base class for model loader. """
    
    def __init__(
            self, model_provider: Optional[Callable], 
            model_kwargs: Dict[str, Any], 
            device_str: str,
    ) -> None:
        self.model_provider = model_provider
        self.model_kwargs = model_kwargs
        self.device_str = device_str

    def load(self) -> Optional[nn.Module]:
        """ Load model. """
        raise NotImplementedError("This method should be overridden by subclasses.")


class ModelLoader(BaseModelLoader):
    """ Model loader to load on GPU/CPU. """
    
    def __init__(
        self, model_provider: Optional[Callable], 
        model_kwargs: Dict[str, Any] = {},
        device_str: str = "cuda:0",
    ) -> None:
        super().__init__(model_provider, model_kwargs, device_str)
        assert device_str == "cpu" or torch.cuda.is_available()
    
    def load(self) -> Optional[nn.Module]:
        """ Load model on GPU/CPU. """
        model = self.model_provider(**self.model_kwargs) if self.model_provider else None
        assert isinstance(model, nn.Module) or model is None
        if model:
            model.to(self.device_str)
        return model
