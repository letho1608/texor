"""Core functionality for Texor native deep learning library"""

# Import native implementations
from .native_tensor import (
    Tensor,
    zeros,
    ones,
    randn,
    tensor as tensor_func
)
from .native_backend import backend

# Legacy autograd for compatibility
from .autograd import backward
from .context import AddBackward, MulBackward, MatMulBackward

# Set device functions
def set_device(device: str) -> None:
    """Set current device"""
    backend.set_device(device)

def get_device() -> str:
    """Get current device"""
    return backend.get_device()

def cuda_is_available() -> bool:
    """Check if CUDA is available"""
    return backend.device_manager.is_gpu_available()

def device_count() -> int:
    """Get number of available devices"""
    return backend.device_manager.device_count() if hasattr(backend.device_manager, 'device_count') else 1

# Additional utility functions
# Additional utility functions
def eye(n: int, m: int = None, dtype=None, device: str = None) -> Tensor:
    """Create identity matrix"""
    import numpy as np
    if m is None:
        m = n
    if device is not None:
        backend.set_device(device)
    data = np.eye(n, m, dtype=dtype or np.float32)
    data = backend.to_device(data)
    return Tensor(data)

def arange(start, stop=None, step=1, dtype=None, device: str = None) -> Tensor:
    """Create range tensor"""
    import numpy as np
    if device is not None:
        backend.set_device(device)
    data = np.arange(start, stop, step, dtype=dtype or np.float32)
    data = backend.to_device(data)
    return Tensor(data)

# Assign the tensor function to avoid module import conflicts
tensor = tensor_func

__all__ = [
    'Tensor',
    'zeros',
    'ones',
    'randn',
    'tensor',
    'eye',
    'arange',
    'backend',
    'set_device',
    'get_device',
    'cuda_is_available',
    'device_count',
    'backward',
    'AddBackward',
    'MulBackward',
    'MatMulBackward'
]