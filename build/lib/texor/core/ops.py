"""Basic operations for Tensor class"""
from typing import Union, Tuple, Optional
import numpy as np
from .tensor import Tensor

def zeros(*shape: Union[int, Tuple[int, ...]]) -> Tensor:
    """Create a tensor filled with zeros"""
    return Tensor(np.zeros(shape))

def ones(*shape: Union[int, Tuple[int, ...]]) -> Tensor:
    """Create a tensor filled with ones"""
    return Tensor(np.ones(shape))

def randn(*shape: Union[int, Tuple[int, ...]]) -> Tensor:
    """Create a tensor filled with random numbers from normal distribution"""
    return Tensor(np.random.randn(*shape))

def zeros_like(tensor: Tensor) -> Tensor:
    """Create a tensor of zeros with the same shape as input"""
    return Tensor(np.zeros_like(tensor.numpy()))

def ones_like(tensor: Tensor) -> Tensor:
    """Create a tensor of ones with the same shape as input"""
    return Tensor(np.ones_like(tensor.numpy()))

def eye(n: int, m: Optional[int] = None) -> Tensor:
    """Create an identity matrix"""
    return Tensor(np.eye(n, m))

def arange(start: int, end: Optional[int] = None, step: int = 1) -> Tensor:
    """Create a 1-D tensor with evenly spaced values"""
    return Tensor(np.arange(start, end, step))

def linspace(start: float, end: float, num: int = 50) -> Tensor:
    """Create a 1-D tensor with evenly spaced values over interval"""
    return Tensor(np.linspace(start, end, num))

def meshgrid(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """Create coordinate matrices from coordinate vectors"""
    grids = np.meshgrid(*[t.numpy() for t in tensors])
    return tuple(Tensor(g) for g in grids)

# Mathematical operations
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication"""
    return a @ b

def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product"""
    return Tensor(np.dot(a.numpy(), b.numpy()))

def transpose(x: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Transpose a tensor"""
    return Tensor(np.transpose(x.numpy(), axes))

def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape a tensor"""
    return Tensor(np.reshape(x.numpy(), shape))

def concat(tensors: Tuple[Tensor, ...], axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis"""
    return Tensor(np.concatenate([t.numpy() for t in tensors], axis=axis))

def stack(tensors: Tuple[Tensor, ...], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis"""
    return Tensor(np.stack([t.numpy() for t in tensors], axis=axis))

def split(x: Tensor, indices_or_sections: Union[int, Tuple[int, ...]], 
         axis: int = 0) -> Tuple[Tensor, ...]:
    """Split a tensor into multiple sub-tensors"""
    parts = np.split(x.numpy(), indices_or_sections, axis=axis)
    return tuple(Tensor(p) for p in parts)

# Statistical operations
def mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute mean along axis"""
    return Tensor(np.mean(x.numpy(), axis=axis, keepdims=keepdims))

def sum(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute sum along axis"""
    return Tensor(np.sum(x.numpy(), axis=axis, keepdims=keepdims))

def max(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute maximum along axis"""
    return Tensor(np.max(x.numpy(), axis=axis, keepdims=keepdims))

def min(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute minimum along axis"""
    return Tensor(np.min(x.numpy(), axis=axis, keepdims=keepdims))

def std(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute standard deviation along axis"""
    return Tensor(np.std(x.numpy(), axis=axis, keepdims=keepdims))

def var(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Compute variance along axis"""
    return Tensor(np.var(x.numpy(), axis=axis, keepdims=keepdims))