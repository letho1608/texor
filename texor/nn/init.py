import numpy as np
from ..core.native_tensor import Tensor

def zeros(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros"""
    tensor.data.fill(0)
    return tensor

def ones(tensor: Tensor) -> Tensor:
    """Fill tensor with ones"""
    tensor.data.fill(1)
    return tensor

def constant(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor with constant value"""
    tensor.data.fill(val)
    return tensor

def normal(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with normal distribution"""
    tensor.data = np.random.normal(mean, std, tensor.shape).astype(tensor.data.dtype)
    return tensor

def uniform(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor with uniform distribution"""
    tensor.data = np.random.uniform(a, b, tensor.shape).astype(tensor.data.dtype)
    return tensor

def xavier_uniform(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) uniform initialization"""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Xavier initialization needs at least 2 dimensions")
    
    fan_in = shape[1] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
    fan_out = shape[0] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
    
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return uniform(tensor, -a, a)

def xavier_normal(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) normal initialization"""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Xavier initialization needs at least 2 dimensions")
    
    fan_in = shape[1] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
    fan_out = shape[0] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
    
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    return normal(tensor, 0.0, std)

def kaiming_uniform(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> Tensor:
    """Kaiming (He) uniform initialization"""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Kaiming initialization needs at least 2 dimensions")
    
    fan_in = shape[1] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
    fan_out = shape[0] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
    
    fan = fan_in if mode == 'fan_in' else fan_out
    
    # Calculate gain
    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a ** 2))
    else:
        gain = 1.0
        
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    return uniform(tensor, -bound, bound)

def kaiming_normal(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> Tensor:
    """Kaiming (He) normal initialization"""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Kaiming initialization needs at least 2 dimensions")
    
    fan_in = shape[1] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
    fan_out = shape[0] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
    
    fan = fan_in if mode == 'fan_in' else fan_out
    
    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a ** 2))
    else:
        gain = 1.0
        
    std = gain / np.sqrt(fan)
    return normal(tensor, 0.0, std)
