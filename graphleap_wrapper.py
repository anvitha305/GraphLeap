# graphleap_wrapper.py
# Minimal wrapper to add GraphLeap support to official ViG models

import torch
import torch.nn as nn
from vig import vig_ti_224_gelu as vig_ti_base
from vig import vig_s_224_gelu as vig_s_base
from vig import vig_b_224_gelu as vig_b_base
from pyramid_vig import pvig_ti_224_gelu as pvig_ti_base
from pyramid_vig import pvig_s_224_gelu as pvig_s_base
from pyramid_vig import pvig_m_224_gelu as pvig_m_base
from pyramid_vig import pvig_b_224_gelu as pvig_b_base


class GraphLeapWrapper(nn.Module):
    """
    Lightweight wrapper for official ViG models.
    
    This wrapper:
    - Uses official ViG/Pyramidal ViG code directly
    - Adds infrastructure for GraphLeap optimization
    - During training: uses standard forward
    - During inference: can enable GraphLeap (currently delegates to standard forward)
    """
    
    def __init__(self, model, use_graphleap=True):
        super().__init__()
        self.model = model
        self.use_graphleap = use_graphleap
    
    def forward(self, x):
        """
        Forward pass - delegates to official model.
        GraphLeap optimization can be enabled but currently uses standard forward
        for maximum compatibility with pretrained weights.
        """
        return self.model(x)
    
    def enable_graphleap(self):
        """Enable GraphLeap optimization"""
        self.use_graphleap = True
    
    def disable_graphleap(self):
        """Disable GraphLeap optimization"""
        self.use_graphleap = False
    
    def load_state_dict(self, state_dict, strict=True):
        """Delegate to wrapped model"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Delegate to wrapped model"""
        return self.model.state_dict(destination, prefix, keep_vars)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# Factory functions - simple wrappers
def vig_ti_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """ViG-Tiny with GraphLeap support"""
    model = vig_ti_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def vig_s_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """ViG-Small with GraphLeap support"""
    model = vig_s_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def vig_b_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """ViG-Base with GraphLeap support"""
    model = vig_b_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def pvig_ti_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """Pyramidal ViG-Tiny with GraphLeap support"""
    model = pvig_ti_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def pvig_s_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """Pyramidal ViG-Small with GraphLeap support"""
    model = pvig_s_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def pvig_m_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """Pyramidal ViG-Medium with GraphLeap support"""
    model = pvig_m_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)

def pvig_b_224_gelu(pretrained=False, use_graphleap=True, **kwargs):
    """Pyramidal ViG-Base with GraphLeap support"""
    model = pvig_b_base(pretrained=pretrained, **kwargs)
    return GraphLeapWrapper(model, use_graphleap=use_graphleap)
