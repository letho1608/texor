"""Functional interface to neural network operations"""
from typing import Optional
from ..core import Tensor
import numpy as np

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Apply dropout to input tensor"""
    if not training or p == 0:
        return x
        
    mask = Tensor(
        np.random.binomial(1, 1-p, x.shape).astype(np.float32)
    )
    return x * mask / (1 - p)

def zeros_like(x: Tensor) -> Tensor:
    """Create a tensor of zeros with the same shape as input"""
    return Tensor(np.zeros_like(x.numpy()))

def ones_like(x: Tensor) -> Tensor:
    """Create a tensor of ones with the same shape as input"""
    return Tensor(np.ones_like(x.numpy()))