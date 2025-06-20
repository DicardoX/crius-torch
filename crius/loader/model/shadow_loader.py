#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Shadow model loader. """

from typing import (Optional, Callable, Dict, Any)

import torch
from torch import nn
from torch._subclasses import FakeTensorMode

from crius.loader.model.loader import BaseModelLoader

__all__ = [
    "ShadowModelLoader",
]


class ShadowModelLoader(BaseModelLoader):
    """ Shadow model loader to instantiate with fake tensors. """

    def __init__(
        self, model_provider: Optional[Callable], 
        model_kwargs: Dict[str, Any] = {},
        fake_device_str: str = "cpu",   # Only simulation
    ) -> None:
        super().__init__(model_provider, model_kwargs, fake_device_str)

        # Shift .device attr of fake model parameters
        torch.__future__.set_swap_module_params_on_conversion(True)
        
    def load(self) -> Optional[nn.Module]:
        """ Load model with fake tensors. """
        
        with FakeTensorMode(allow_non_fake_inputs=True) as global_fake_mode:
            model = self.model_provider(**self.model_kwargs) if self.model_provider else None
            assert isinstance(model, nn.Module) or model is None
            if model:
                model.to(self.device_str)
            
            # Fake tensor check
            for param in model.parameters():
                assert isinstance(param, torch._subclasses.FakeTensor), \
                    f"Parameter {param.__class__} is not a FakeTensor."
            
            return model
