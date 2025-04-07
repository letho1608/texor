"""Context classes for autograd operations"""
from typing import List, Union, Tuple
import numpy as np

class Context:
    """Context for storing information needed for backpropagation"""
    def __init__(self, inputs: List['Tensor']):
        self.inputs = inputs
        self.saved_tensors: List['Tensor'] = []
        
    def save_for_backward(self, *tensors: 'Tensor') -> None:
        """Save tensors needed for backward pass"""
        self.saved_tensors = list(tensors)  # Replace list instead of extending
        
    def backward(self, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Compute gradients with respect to inputs"""
        raise NotImplementedError

class AddBackward(Context):
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output
class MulBackward(Context):
    def __init__(self, inputs: List['Tensor']):
        super().__init__(inputs)
        self.save_for_backward(*inputs)
        
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input1, input2 = self.saved_tensors
        print(f"\nMulBackward grad_output: {grad_output}")
        print(f"Input tensors: {input1.numpy()}, {input2.numpy()}")
        grad1 = grad_output * input2.numpy()
        grad2 = grad_output * input1.numpy()
        print(f"Computed grads: {grad1}, {grad2}")
        return grad1, grad2

class MatMulBackward(Context):
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input1, input2 = self.inputs
        return (grad_output @ input2.numpy().T,
                input1.numpy().T @ grad_output)

class DivBackward(Context):
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input1, input2 = self.inputs
        input1_data = input1.numpy()
        input2_data = input2.numpy()
        return (grad_output / input2_data,
                -grad_output * input1_data / (input2_data * input2_data))

class SumBackward(Context):
    def __init__(self, inputs: List['Tensor'], axis: Union[int, None], keepdims: bool):
        super().__init__(inputs)
        self.axis = axis
        self.keepdims = keepdims
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        input_shape = self.inputs[0].shape
        print(f"\nSumBackward grad_output: {grad_output}")
        print(f"Input shape: {input_shape}")
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        result = np.broadcast_to(grad_output, input_shape)
        print(f"Broadcasted grad: {result}")
        return result

# Import Tensor at the end to avoid circular import
from .tensor import Tensor