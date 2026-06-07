"""
Native tensor implementation for Texor
Combines PyTorch-style dynamic computation with TensorFlow-style optimizations
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
from .native_backend import backend


class GradientFunction:
    """Base class for gradient functions (inspired by PyTorch autograd)"""
    
    def __init__(self, *inputs: 'Tensor'):
        self.inputs = inputs
        self.needs_input_grad = [x.requires_grad for x in inputs if isinstance(x, Tensor)]
        
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        """Compute gradients with respect to inputs"""
        raise NotImplementedError


class AddBackward(GradientFunction):
    """Gradient function for addition"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        a, b = self.inputs
        grad_a = grad_output if self.needs_input_grad[0] else None
        grad_b = grad_output if self.needs_input_grad[1] else None
        
        # Handle broadcasting for bias terms
        if grad_a is not None and grad_a.shape != a.shape:
            # Sum over broadcasted dimensions
            grad_a = self._reduce_gradient(grad_a, a.shape)
        
        if grad_b is not None and grad_b.shape != b.shape:
            # Sum over broadcasted dimensions
            grad_b = self._reduce_gradient(grad_b, b.shape)
            
        return grad_a, grad_b
    
    def _reduce_gradient(self, grad: 'Tensor', target_shape: Tuple[int, ...]) -> 'Tensor':
        """Reduce gradient to match target shape"""
        # Sum over axes that were broadcasted
        ndim_added = grad.data.ndim - len(target_shape)
        for i in range(ndim_added):
            grad = Tensor(np.sum(grad.data, axis=0), requires_grad=False)
        
        # Sum over axes that are size 1 in target but not in grad
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                grad = Tensor(np.sum(grad.data, axis=i, keepdims=True), requires_grad=False)
                
        return grad


class MulBackward(GradientFunction):
    """Gradient function for element-wise multiplication"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        a, b = self.inputs
        grad_a = grad_output * b if self.needs_input_grad[0] else None
        grad_b = grad_output * a if self.needs_input_grad[1] else None
        return grad_a, grad_b


class MatMulBackward(GradientFunction):
    """Gradient function for matrix multiplication"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        a, b = self.inputs
        grad_a = grad_output @ b.T if self.needs_input_grad[0] else None
        grad_b = a.T @ grad_output if self.needs_input_grad[1] else None
        return grad_a, grad_b


class Conv2DBackward(GradientFunction):
    """Gradient function for 2D convolution"""
    
    def __init__(self, input_tensor, weight, stride=1, padding=0):
        super().__init__(input_tensor, weight)
        self.stride = stride
        self.padding = padding
        
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor, weight = self.inputs
        stride = self.stride
        padding = self.padding
        
        # Format padding
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding
            
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
            
        N, C_in, H_in, W_in = input_tensor.shape
        C_out, _, kH, kW = weight.shape
        _, _, H_out, W_out = grad_output.shape
        
        # Apply padding to input
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(input_tensor.data, 
                              ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 
                              mode='constant')
        else:
            x_padded = input_tensor.data
            
        grad_input = None
        if self.needs_input_grad[0]:
            grad_x_padded = np.zeros_like(x_padded)
            # Compute gradient w.r.t input
            # dx = grad_output * weight (transposed)
            for n in range(N):
                for c_out in range(C_out):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride_h
                            w_start = w * stride_w
                            grad_x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW] += \
                                weight.data[c_out] * grad_output.data[n, c_out, h, w]
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                grad_x = grad_x_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            else:
                grad_x = grad_x_padded
            grad_input = Tensor(grad_x)
            
        grad_weight = None
        if self.needs_input_grad[1]:
            grad_w = np.zeros_like(weight.data)
            # Compute gradient w.r.t weight
            # dw = x * grad_output
            for n in range(N):
                for c_out in range(C_out):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride_h
                            w_start = w * stride_w
                            patch = x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                            grad_w[c_out] += patch * grad_output.data[n, c_out, h, w]
            grad_weight = Tensor(grad_w)
            
        return grad_input, grad_weight


class ReLUBackward(GradientFunction):
    """Gradient function for ReLU"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        mask = input_tensor.data > 0
        grad_input = Tensor(grad_output.data * mask) if self.needs_input_grad[0] else None
        return (grad_input,)


class PowerBackward(GradientFunction):
    """Gradient function for power operations"""
    
    def __init__(self, input_tensor: 'Tensor', power: float):
        super().__init__(input_tensor)
        self.power = power
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if self.needs_input_grad[0]:
            # d/dx (x^p) = p * x^(p-1)
            grad_input = grad_output * (self.power * (input_tensor.data ** (self.power - 1)))
            grad_input = Tensor(grad_input.data)
        else:
            grad_input = None
        return (grad_input,)


class ExpBackward(GradientFunction):
    """Gradient function for exponential"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if self.needs_input_grad[0]:
            # d/dx (e^x) = e^x
            exp_result = backend.exp(input_tensor.data)
            grad_input = Tensor(grad_output.data * exp_result)
        else:
            grad_input = None
        return (grad_input,)


class LogBackward(GradientFunction):
    """Gradient function for natural logarithm"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if self.needs_input_grad[0]:
            # d/dx (ln(x)) = 1/x
            grad_input = Tensor(grad_output.data / input_tensor.data)
        else:
            grad_input = None
        return (grad_input,)


class NegBackward(GradientFunction):
    """Gradient function for negation"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        return (-grad_output,) if self.needs_input_grad[0] else (None,)


class SubBackward(GradientFunction):
    """Gradient function for subtraction"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        a, b = self.inputs
        grad_a = grad_output if self.needs_input_grad[0] else None
        grad_b = (-grad_output) if self.needs_input_grad[1] else None
        
        # Handle broadcasting
        if grad_a is not None and grad_a.shape != a.shape:
            grad_a = self._reduce_gradient(grad_a, a.shape)
        if grad_b is not None and grad_b.shape != b.shape:
            grad_b = self._reduce_gradient(grad_b, b.shape)
            
        return grad_a, grad_b

    def _reduce_gradient(self, grad: 'Tensor', target_shape: Tuple[int, ...]) -> 'Tensor':
        """Reduce gradient to match target shape"""
        # Sum over axes that were broadcasted
        ndim_added = grad.data.ndim - len(target_shape)
        for i in range(ndim_added):
            grad = Tensor(np.sum(grad.data, axis=0), requires_grad=False)
        
        # Sum over axes that are size 1 in target but not in grad
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                grad = Tensor(np.sum(grad.data, axis=i, keepdims=True), requires_grad=False)
                
        return grad


class DivBackward(GradientFunction):
    """Gradient function for division"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        a, b = self.inputs
        # d/da (a/b) = 1/b
        grad_a = grad_output / b if self.needs_input_grad[0] else None
        # d/db (a/b) = -a/b^2
        grad_b = grad_output * (-a / (b * b)) if self.needs_input_grad[1] else None
        
        # Handle broadcasting
        if grad_a is not None and grad_a.shape != a.shape:
            grad_a = self._reduce_gradient(grad_a, a.shape)
        if grad_b is not None and grad_b.shape != b.shape:
            grad_b = self._reduce_gradient(grad_b, b.shape)
            
        return grad_a, grad_b

    def _reduce_gradient(self, grad: 'Tensor', target_shape: Tuple[int, ...]) -> 'Tensor':
        """Reduce gradient to match target shape"""
        # Sum over axes that were broadcasted
        ndim_added = grad.data.ndim - len(target_shape)
        for i in range(ndim_added):
            grad = Tensor(np.sum(grad.data, axis=0), requires_grad=False)
        
        # Sum over axes that are size 1 in target but not in grad
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                grad = Tensor(np.sum(grad.data, axis=i, keepdims=True), requires_grad=False)
                
        return grad


class ReshapeBackward(GradientFunction):
    """Gradient function for reshape operations"""
    
    def __init__(self, input_tensor: 'Tensor', original_shape: Tuple[int, ...]):
        super().__init__(input_tensor)
        self.original_shape = original_shape
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        if not self.needs_input_grad[0]:
            return None,
        return Tensor(grad_output.data.reshape(self.original_shape)),


class TransposeBackward(GradientFunction):
    """Gradient function for transpose operations"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        if not self.needs_input_grad[0]:
            return None,
        return Tensor(grad_output.data.T),


class SigmoidBackward(GradientFunction):
    """Gradient function for sigmoid"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if not self.needs_input_grad[0]:
            return None,
        
        # s = 1 / (1 + exp(-x))
        # ds/dx = s * (1 - s)
        s_data = backend.sigmoid(input_tensor.data)
        grad_input = grad_output.data * s_data * (1 - s_data)
        return Tensor(grad_input),


class TanhBackward(GradientFunction):
    """Gradient function for tanh"""
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if not self.needs_input_grad[0]:
            return None,
        
        # t = tanh(x)
        # dt/dx = 1 - t^2
        t_data = backend.tanh(input_tensor.data)
        grad_input = grad_output.data * (1 - t_data**2)
        return Tensor(grad_input),


class SoftmaxBackward(GradientFunction):
    """Gradient function for softmax"""
    
    def __init__(self, input_tensor: 'Tensor', axis: int):
        super().__init__(input_tensor)
        self.axis = axis
        
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        input_tensor = self.inputs[0]
        if not self.needs_input_grad[0]:
            return None,
            
        # s = softmax(x)
        # ds/dx = s * (1 - s) for same index, -s_i * s_j for different indices
        # Simplified vector-form: grad_input = s * (grad_output - sum(grad_output * s))
        s_data = backend.softmax(input_tensor.data, axis=self.axis)
        grad_input = s_data * (grad_output.data - np.sum(grad_output.data * s_data, axis=self.axis, keepdims=True))
        
        return Tensor(grad_input),


class SumBackward(GradientFunction):
    """Gradient function for sum operations"""
    
    def __init__(self, input_tensor: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool):
        super().__init__(input_tensor)
        self.axis = axis
        self.keepdims = keepdims
        self.original_shape = input_tensor.shape
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        if not self.needs_input_grad[0]:
            return None,
        
        # Expand dimensions that were reduced
        grad = grad_output.data
        if not self.keepdims and self.axis is not None:
            if isinstance(self.axis, int):
                grad = np.expand_dims(grad, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad = np.expand_dims(grad, ax)
        
        # Broadcast to original shape
        grad = np.broadcast_to(grad, self.original_shape)
        return Tensor(grad, requires_grad=False),


class MeanBackward(GradientFunction):
    """Gradient function for mean operations"""
    
    def __init__(self, input_tensor: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool):
        super().__init__(input_tensor)
        self.axis = axis
        self.keepdims = keepdims
        self.original_shape = input_tensor.shape
        
        # Calculate the number of elements averaged over
        if axis is None:
            self.num_elements = input_tensor.data.size
        else:
            if isinstance(axis, int):
                self.num_elements = input_tensor.shape[axis]
            else:
                self.num_elements = 1
                for ax in axis:
                    self.num_elements *= input_tensor.shape[ax]
    
    def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
        if not self.needs_input_grad[0]:
            return None,
        
        # Expand dimensions that were reduced
        grad = grad_output.data
        if not self.keepdims and self.axis is not None:
            if isinstance(self.axis, int):
                grad = np.expand_dims(grad, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad = np.expand_dims(grad, ax)
        
        # Broadcast to original shape
        grad = np.broadcast_to(grad, self.original_shape)
        
        # Divide by number of elements (mean gradient property)
        grad = grad / self.num_elements
        
        return Tensor(grad, requires_grad=False),


class Tensor:
    """
    Native tensor implementation combining best practices from PyTorch and TensorFlow
    - Dynamic computation graph like PyTorch
    - Optimized operations like TensorFlow
    - Device management
    - Automatic differentiation
    """
    
    def __init__(self, data: Any, requires_grad: bool = False, device: Optional[str] = None):
        """Initialize tensor"""
        
        # Convert input data to numpy array
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32) if data.dtype != np.float32 else data.copy()
        else:
            self.data = np.array(data, dtype=np.float32)
        
        # Move to specified device
        if device is not None:
            backend.set_device(device)
        self.data = backend.to_device(self.data)
        
        # Gradient tracking
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._version = 0
        
        # Computation graph tracking
        self._backward_hooks = []
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get tensor dtype"""
        return self.data.dtype
    
    def __len__(self) -> int:
        """Get length of first dimension"""
        if len(self.shape) == 0:
            return 1
        return self.shape[0]
    
    @property
    def device(self) -> str:
        """Get tensor device"""
        return backend.get_device()
    
    @property
    def T(self) -> 'Tensor':
        """Transpose (2D only)"""
        if len(self.shape) != 2:
            raise ValueError("Transpose only supported for 2D tensors")
        return Tensor(self.data.T, requires_grad=self.requires_grad)
    
    def to(self, device: str) -> 'Tensor':
        """Move tensor to device"""
        new_data = backend.to_device(self.data, device)
        return Tensor(new_data, requires_grad=self.requires_grad)
    
    def detach(self) -> 'Tensor':
        """Detach from computation graph"""
        return Tensor(self.data, requires_grad=False)
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        if backend.get_device() == 'cuda':
            try:
                import cupy as cp
                if isinstance(self.data, cp.ndarray):
                    return cp.asnumpy(self.data)
            except ImportError:
                pass
        return self.data
    
    def __array__(self, dtype=None) -> np.ndarray:
        """NumPy array interface"""
        if dtype:
            return self.numpy().astype(dtype)
        return self.numpy()
    
    def item(self) -> float:
        """Get scalar value"""
        if self.data.size != 1:
            raise ValueError("Only one element tensors can be converted to Python scalars")
        return float(self.numpy().item())
    
    def zero_grad(self) -> None:
        """Zero out gradients"""
        if self.grad is not None:
            self.grad.data.fill(0)
    
    def backward(self, gradient: Optional['Tensor'] = None, retain_graph: bool = False) -> None:
        """Compute gradients via backpropagation"""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")
        
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = Tensor(np.ones_like(self.data))
        
        # Initialize gradient if not exists
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))
        
        # Accumulate gradient
        self.grad.data += gradient.data
          # Backpropagate through computation graph
        if self.grad_fn is not None:
            input_grads = self.grad_fn.backward(gradient)
            
            for input_tensor, input_grad in zip(self.grad_fn.inputs, input_grads):
                if input_grad is not None and isinstance(input_tensor, Tensor) and input_tensor.requires_grad:
                    input_tensor.backward(input_grad, retain_graph=retain_graph)
    
    # Mathematical operations
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Addition"""
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        elif not isinstance(other, Tensor):
            other = Tensor(other)
        
        result_data = backend.add(self.data, other.data)
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = AddBackward(self, other)
        
        return result
    
    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Subtraction"""
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        elif not isinstance(other, Tensor):
            other = Tensor(other)
            
        result_data = self.data - other.data
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SubBackward(self, other)
            
        return result

    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other - self

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Division"""
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        elif not isinstance(other, Tensor):
            other = Tensor(other)
            
        result_data = self.data / other.data
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = DivBackward(self, other)
            
        return result

    def __rtruediv__(self, other: Union[float, int]) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other / self
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication"""
        if isinstance(other, (int, float)):
            other_tensor = Tensor(np.full_like(self.data, other))
            result_data = self.data * other
        elif isinstance(other, Tensor):
            other_tensor = other
            result_data = self.data * other.data
        else:
            other_tensor = Tensor(other)
            result_data = self.data * other_tensor.data
        
        requires_grad = self.requires_grad or other_tensor.requires_grad
        result = Tensor(result_data, requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = MulBackward(self, other_tensor)
        
        return result
    
    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        return self.__mul__(other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result_data = backend.matmul(self.data, other.data)
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = MatMulBackward(self, other)
        
        return result
    
    def __neg__(self) -> 'Tensor':
        """Negation"""
        result = Tensor(-self.data, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.grad_fn = NegBackward(self)
        return result
    
    def __getitem__(self, key) -> 'Tensor':
        """Indexing"""
        return Tensor(self.data[key], requires_grad=self.requires_grad)
    
    def abs(self) -> 'Tensor':
        """Absolute value"""
        result_data = np.abs(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        # Simplified: for abs(x), grad is sign(x)
        if result.requires_grad:
            class AbsBackward(GradientFunction):
                def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
                    input_tensor = self.inputs[0]
                    return (grad_output * np.sign(input_tensor.data),)
            result.grad_fn = AbsBackward(self)
        return result

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum along axis"""
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SumBackward(self, axis, keepdims)
        
        return result
        
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Mean along axis"""
        result_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = MeanBackward(self, axis, keepdims)
            
        return result
    
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Reshape tensor"""
        result_data = self.data.reshape(shape)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.grad_fn = ReshapeBackward(self, self.shape)
        return result
    
    def view(self, shape: Tuple[int, ...]) -> 'Tensor':
        """View tensor with new shape (alias for reshape)"""
        return self.reshape(shape)
    
    def transpose(self) -> 'Tensor':
        """Transpose (2D only)"""
        if len(self.shape) != 2:
            raise ValueError("Transpose only supported for 2D tensors")
        result = Tensor(self.data.T, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.grad_fn = TransposeBackward(self)
        return result

    def relu(self) -> 'Tensor':
        """ReLU activation"""
        result_data = backend.relu(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = ReLUBackward(self)
        
        return result
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation"""
        result_data = backend.sigmoid(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SigmoidBackward(self)
        
        return result

    def tanh(self) -> 'Tensor':
        """Tanh activation"""
        result_data = backend.tanh(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = TanhBackward(self)
        
        return result
    
    def pow(self, power: float) -> 'Tensor':
        """Element-wise power operation"""
        result_data = backend.power(self.data, power)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = PowerBackward(self, power)
        
        return result
    
    def sqrt(self) -> 'Tensor':
        """Element-wise square root"""
        return self.pow(0.5)
    
    def exp(self) -> 'Tensor':
        """Element-wise exponential"""
        result_data = backend.exp(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = ExpBackward(self)
        
        return result
    
    def log(self) -> 'Tensor':
        """Element-wise natural logarithm"""
        result_data = backend.log(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = LogBackward(self)
        
        return result

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax activation"""
        result_data = backend.softmax(self.data, axis=axis)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SoftmaxBackward(self, axis)
            
        return result
    
    def broadcast_to(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Broadcast tensor to new shape"""
        if self.shape == shape:
            return self
            
        result_data = np.broadcast_to(self.data, shape)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            class BroadcastBackward(GradientFunction):
                def __init__(self, input_tensor: 'Tensor'):
                    super().__init__(input_tensor)
                    self.original_shape = input_tensor.shape
                
                def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
                    if not self.needs_input_grad[0]:
                        return None,
                    # Reducing broadcast is a sum
                    grad = grad_output.data
                    ndim_added = grad.ndim - len(self.original_shape)
                    for _ in range(ndim_added):
                        grad = np.sum(grad, axis=0)
                    
                    for i, (dim, orig_dim) in enumerate(zip(grad.shape, self.original_shape)):
                        if orig_dim == 1 and dim > 1:
                            grad = np.sum(grad, axis=i, keepdims=True)
                            
                    return Tensor(grad),
            
            result.grad_fn = BroadcastBackward(self)
            
        return result

    def gather(self, axis: int, index: 'Tensor') -> 'Tensor':
        """Gather values along an axis specified by index"""
        result_data = backend.gather(self.data, index.data.astype(int), axis=axis)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            class GatherBackward(GradientFunction):
                def __init__(self, input_tensor: 'Tensor', axis: int, index: 'Tensor'):
                    super().__init__(input_tensor, index)
                    self.axis = axis
                    self.index = index
                
                def backward(self, grad_output: 'Tensor') -> Tuple[Optional['Tensor'], ...]:
                    input_tensor = self.inputs[0]
                    if not self.needs_input_grad[0]:
                        return None, None
                    
                    # Gradient of gather is scatter_add
                    grad_input = np.zeros_like(input_tensor.data)
                    np.add.at(grad_input, (slice(None),) * self.axis + (self.index.data.astype(int),), grad_output.data)
                    return Tensor(grad_input), None
            
            result.grad_fn = GatherBackward(self, axis, index)
            
        return result
    
    def __repr__(self) -> str:
        """String representation"""
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        device_str = f", device='{self.device}'" if self.device != 'cpu' else ""
        return f"Tensor({self.data!r}{grad_str}{device_str})"


# Utility functions
def zeros(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None) -> Tensor:
    """Create zero tensor"""
    if device is not None:
        backend.set_device(device)
    data = backend.zeros(shape)
    return Tensor(data, requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None) -> Tensor:
    """Create ones tensor"""
    if device is not None:
        backend.set_device(device)
    data = backend.ones(shape)
    return Tensor(data, requires_grad=requires_grad)


def randn(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None) -> Tensor:
    """Create random normal tensor"""
    if device is not None:
        backend.set_device(device)
    data = backend.randn(shape)
    return Tensor(data, requires_grad=requires_grad)


def tensor(data: Any, requires_grad: bool = False, device: Optional[str] = None) -> Tensor:
    """Create tensor from data"""
    return Tensor(data, requires_grad=requires_grad, device=device)