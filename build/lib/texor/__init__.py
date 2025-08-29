"""
Texor - Native Deep Learning Framework

A lightweight, PyTorch-style deep learning framework built from scratch in Python.
Provides tensor operations, automatic differentiation, neural networks, and optimizers
without heavy dependencies.

Example:
    >>> import texor
    >>> from texor.core import randn
    >>> from texor.nn import Sequential, Linear, ReLU
    >>> 
    >>> # Create model
    >>> model = Sequential([
    ...     Linear(784, 128),
    ...     ReLU(),
    ...     Linear(128, 10)
    ... ])
    >>> 
    >>> # Forward pass
    >>> x = randn((32, 784))
    >>> output = model(x)
"""

# Import core functionality
from .core import (
    Tensor,
    zeros,
    ones,
    randn,
    tensor,
    eye,
    arange,
    set_device,
    get_device,
    cuda_is_available,
    device_count
)

# Import neural network components (commonly used)
from .nn import (
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    MSELoss,
    CrossEntropyLoss
)

# Import optimizers (commonly used)
from .optim import Adam, SGD

# Version
from .version import __version__

# Set default device
set_device('cpu')

# Configure numpy print options for better tensor display
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=1000)

__all__ = [
    # Core
    'Tensor', 'zeros', 'ones', 'randn', 'tensor', 'eye', 'arange',
    'set_device', 'get_device', 'cuda_is_available', 'device_count',
    
    # Neural Networks (commonly used)
    'Sequential', 'Linear', 'ReLU', 'Sigmoid',
    'MSELoss', 'CrossEntropyLoss',
    
    # Optimizers (commonly used)
    'Adam', 'SGD',
    
    # Version
    '__version__'
]

# Framework information
def info():
    """Print Texor framework information"""
    from rich.console import Console
    from rich.panel import Panel
    import platform
    
    console = Console()
    console.print(Panel(
        f"[bold blue]Texor v{__version__}[/bold blue] - Native Deep Learning Framework\n" +
        "[dim]Lightweight ML library with PyTorch-style API[/dim]\n\n" +
        f"[yellow]Python:[/yellow] {platform.python_version()}\n" +
        f"[yellow]Platform:[/yellow] {platform.platform()}\n" +
        f"[yellow]Device:[/yellow] {get_device()}\n" +
        f"[yellow]GPU Available:[/yellow] {cuda_is_available()}",
        title="Framework Info",
        style="green"
    ))

# Configure warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')