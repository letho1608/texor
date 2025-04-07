"""Core functionality for Nexor deep learning library"""

from .tensor import Tensor
from .autograd import backward
from .context import AddBackward, MulBackward, MatMulBackward
from .ops import (
    zeros,
    ones,
    randn,
    zeros_like,
    ones_like,
    eye,
    arange,
    linspace,
    meshgrid,
    matmul,
    dot,
    transpose,
    reshape,
    concat,
    stack,
    split,
    mean,
    sum,
    max,
    min,
    std,
    var
)
from .backend import backend

__all__ = [
    # Main classes
    'Tensor',
    'backend',
    
    # Autograd
    'backward',
    'AddBackward',
    'MulBackward',
    'MatMulBackward',
    
    # Creation ops
    'zeros',
    'ones',
    'randn',
    'zeros_like',
    'ones_like',
    'eye',
    'arange',
    'linspace',
    'meshgrid',
    
    # Math ops
    'matmul',
    'dot',
    'transpose',
    'reshape',
    'concat',
    'stack',
    'split',
    
    # Statistical ops
    'mean',
    'sum',
    'max',
    'min',
    'std',
    'var'
]

# Set backend to auto by default
backend.set_backend('auto')