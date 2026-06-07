from typing import Optional
import numpy as np
from ..core.native_tensor import Tensor
from ..core.native_backend import backend

class Activation:
    """Base class for all activation functions"""
    
    def __init__(self):
        self.trainable: bool = False
        
    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)
        
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ReLU(Activation):
    """Rectified Linear Unit activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        # Use native tensor's built-in ReLU
        return inputs.relu()

class Sigmoid(Activation):
    """Sigmoid activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.sigmoid()

class Tanh(Activation):
    """Hyperbolic tangent activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()

class LeakyReLU(Activation):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in range [0, 1)")
        self.alpha = alpha
        
    def forward(self, inputs: Tensor) -> Tensor:
        # Simplified: use Tensor operations to preserve graph
        # alpha * x if x < 0 else x
        # This is a bit tricky without a clamp or where in Tensor
        # But we can use: relu(x) - alpha * relu(-x)
        return inputs.relu() - self.alpha * ((-inputs).relu())

class ELU(Activation):
    """Exponential Linear Unit activation function"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        
    def forward(self, inputs: Tensor) -> Tensor:
        # ELU(x) = alpha * (exp(x) - 1) if x < 0 else x
        # Again, tricky without where. 
        # For now, let's use a simplified version if possible or just the data-based one if we must,
        # but data-based one breaks the graph.
        # Let's try to use relu and exp.
        # ELU(x) = relu(x) + min(0, alpha * (exp(x) - 1))
        # Since x < 0 => exp(x) - 1 < 0
        return inputs.relu() - self.alpha * (1 - inputs.exp()).relu()

class Softmax(Activation):
    """Softmax activation function"""
    
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
        
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.softmax(axis=self.axis)


class GELU(Activation):
    """Gaussian Error Linear Unit activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.gelu(inputs)
        x = inputs.numpy()
        return Tensor(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        x = self.cached_input.numpy()
        # Approximate GELU gradient
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
        return grad * (cdf + x * pdf)

# Factory function
def get_activation(name: str) -> Activation:
    """Get activation function by name"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'softmax': Softmax,
        'gelu': GELU
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
        
    return activations[name]()