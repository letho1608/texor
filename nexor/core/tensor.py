import numpy as np
import tensorflow as tf
import torch
from typing import TYPE_CHECKING, Union, Tuple, Optional

if TYPE_CHECKING:
    from .context import Context, AddBackward, MulBackward, MatMulBackward
    
class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Initialize a Tensor object that can leverage both TF and PyTorch backends.
        
        Args:
            data: Can be numpy array, TF tensor, PyTorch tensor, or Python list/tuple
            requires_grad: Boolean indicating if we need to compute gradients
        """
        # Convert data to numpy array and ensure float type for gradient computation
        if isinstance(data, np.ndarray):
            self._numpy = data.astype(np.float32) if data.dtype.kind in 'iu' else data
        elif isinstance(data, tf.Tensor):
            self._numpy = data.numpy().astype(np.float32) if data.dtype.is_integer else data.numpy()
        elif isinstance(data, torch.Tensor):
            self._numpy = data.detach().cpu().numpy().astype(np.float32) if data.dtype in [torch.int32, torch.int64] else data.detach().cpu().numpy()
        else:
            self._numpy = np.array(data, dtype=np.float32)
            
        self._tf_tensor = tf.convert_to_tensor(self._numpy)
        self._torch_tensor = torch.from_numpy(self._numpy)
        self.requires_grad = requires_grad
        self._ctx = None
        self._grad = None
        
        if requires_grad:
            self._torch_tensor.requires_grad_(True)
            
    @property
    def shape(self):
        return self._numpy.shape
        
    @property
    def dtype(self):
        return self._numpy.dtype
        
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """Sum elements along given axis"""
        if axis is None:
            result = Tensor(np.sum(self._numpy), requires_grad=self.requires_grad)
        else:
            result = Tensor(np.sum(self._numpy, axis=axis, keepdims=keepdims),
                          requires_grad=self.requires_grad)
            
        if self.requires_grad:
            from .context import SumBackward
            result._ctx = SumBackward([self], axis, keepdims)
            
        return result

    def numpy(self):
        return self._numpy
        
    def tensorflow(self):
        return self._tf_tensor
        
    def pytorch(self):
        return self._torch_tensor
        
    def __repr__(self):
        return f"Nexor Tensor(shape={self.shape}, dtype={self.dtype})"
        
    # Basic arithmetic operations
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self._numpy + other._numpy,
                       requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            # Import here to avoid circular import
            from .context import AddBackward
            result._ctx = AddBackward([self, other])
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Compute result
        result = Tensor(self._numpy * other._numpy,
                       requires_grad=(self.requires_grad or other.requires_grad))
        
        # Set up backward context if gradient is needed
        if self.requires_grad or other.requires_grad:
            # Import here to avoid circular import
            from .context import MulBackward
            result._ctx = MulBackward([self, other])
            result._ctx.save_for_backward(self, other)
            print(f"\nCreated MulBackward context")
            print(f"Input values: {self.numpy()}, {other.numpy()}")
            print(f"Result requires grad: {result.requires_grad}")
        
        return result

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self._numpy @ other._numpy,
                       requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            # Import here to avoid circular import
            from .context import MatMulBackward
            result._ctx = MatMulBackward([self, other])
        return result

    @property
    def grad(self):
        """Get gradient as numpy array"""
        return self._grad.numpy() if self._grad is not None else None

    def backward(self, gradient: Optional['Tensor'] = None):
        """Compute gradients through this tensor"""
        # Import here to avoid circular import
        from .autograd import backward

        if gradient is None:
            # If no gradient is provided, start with ones
            gradient = Tensor(np.ones_like(self._numpy), requires_grad=False)
            
        backward(self, gradient)

# Factory functions
def zeros(*shape):
    return Tensor(np.zeros(shape))

def ones(*shape):
    return Tensor(np.ones(shape))

def randn(*shape):
    return Tensor(np.random.randn(*shape))

def from_numpy(array):
    return Tensor(array)

def from_tensorflow(tensor):
    return Tensor(tensor)

def from_pytorch(tensor):
    return Tensor(tensor)