from typing import Optional
import numpy as np
from ..core import Tensor
from ..core.backend import backend

class Activation:
    """Base class for all activation functions"""
    
    def __init__(self):
        self.trainable: bool = False
        self.cached_input: Optional[Tensor] = None
        
    def __call__(self, inputs: Tensor) -> Tensor:
        self.cached_input = inputs
        return self.forward(inputs)
        
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
        
    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ReLU(Activation):
    """Rectified Linear Unit activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.relu(inputs)
        return Tensor(np.maximum(0, inputs.numpy()))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        return grad * (self.cached_input.numpy() > 0)

class Sigmoid(Activation):
    """Sigmoid activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.sigmoid(inputs)
        return Tensor(1 / (1 + np.exp(-inputs.numpy())))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        sigmoid = 1 / (1 + np.exp(-self.cached_input.numpy()))
        return grad * sigmoid * (1 - sigmoid)

class Tanh(Activation):
    """Hyperbolic tangent activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.tanh(inputs)
        return Tensor(np.tanh(inputs.numpy()))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        tanh = np.tanh(self.cached_input.numpy())
        return grad * (1 - tanh ** 2)

class LeakyReLU(Activation):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in range [0, 1)")
        self.alpha = alpha
        
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.leaky_relu(inputs, self.alpha)
        x = inputs.numpy()
        return Tensor(np.where(x > 0, x, self.alpha * x))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        x = self.cached_input.numpy()
        dx = np.where(x > 0, 1, self.alpha)
        return grad * dx
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"

class ELU(Activation):
    """Exponential Linear Unit activation function"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.elu(inputs, self.alpha)
        x = inputs.numpy()
        return Tensor(np.where(x > 0, x, self.alpha * (np.exp(x) - 1)))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        x = self.cached_input.numpy()
        dx = np.where(x > 0, 1, self.alpha * np.exp(x))
        return grad * dx
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"

class Softmax(Activation):
    """Softmax activation function"""
    
    def forward(self, inputs: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.softmax(inputs)
        x = inputs.numpy()
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=-1, keepdims=True))
        
    def backward(self, grad: Tensor) -> Tensor:
        if self.cached_input is None:
            raise RuntimeError("Backward called before forward!")
        # Gradient of softmax is more complex and usually combined with cross-entropy loss
        # for numerical stability. See CrossEntropyLoss implementation.
        raise NotImplementedError("Softmax backward should be handled by CrossEntropyLoss")

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