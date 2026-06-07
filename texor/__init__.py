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

# Import neural network components
from .nn import (
    # Layers
    Layer,
    Linear,
    Conv2D,
    Conv1D,
    Conv3D,
    ConvTranspose2D,
    MaxPool2D,
    MaxPool3D,
    AvgPool3D,
    BatchNorm2D,
    LayerNorm,
    GroupNorm,
    InstanceNorm1D,
    InstanceNorm2D,
    Dropout,
    Dropout2D,
    Dropout3D,
    AdaptiveAvgPool2D,
    Embedding,
    Flatten,
    Reshape,
    Sequential,
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    
    # Activations
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Softmax,
    GELU,
    
    # Loss functions
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    L1Loss,
    HuberLoss,
    SmoothL1Loss,
    KLDivLoss,
    
    # Functional
    F,
)

# Import optimizers
from .optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    NAdam,
    RAdam,
    Adamax,
    ASGD,
    LBFGS,
    get_optimizer,
    
    # Schedulers
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    WarmupScheduler,
    CyclicLR,
    OneCycleLR,
)

# Import data utilities
from .data import (
    Dataset,
    TensorDataset,
    ArrayDataset,
    DataLoader,
    SubsetDataset,
    MappedDataset,
    random_split,
    Transform,
    Compose,
    ToTensor,
    ToPILImage,
    ToNumpy,
    ToDtype,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    Resize,
    RandomAffine,
    RandomPerspective,
    ColorJitter,
    RandomErasing,
    RandomChoice,
    RandomApply,
    Lambda,
)

# Import metrics
from .metrics import (
    Metric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MSE,
    MAE,
    RMSE,
    R2Score,
    ConfusionMatrix,
    AUC,
    IoU,
    DiceCoefficient,
    Perplexity,
    MetricCollection,
    get_metric,
)

# Import utils
from . import utils

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
    
    # Neural Networks
    'Layer',
    'Linear',
    'Conv2D',
    'Conv1D',
    'Conv3D',
    'ConvTranspose2D',
    'MaxPool2D',
    'MaxPool3D',
    'AvgPool3D',
    'BatchNorm2D',
    'LayerNorm',
    'GroupNorm',
    'InstanceNorm1D',
    'InstanceNorm2D',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'AdaptiveAvgPool2D',
    'Embedding',
    'Flatten',
    'Reshape',
    'Sequential',
    'MultiheadAttention',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'LeakyReLU',
    'ELU',
    'Softmax',
    'GELU',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    'HuberLoss',
    'SmoothL1Loss',
    'KLDivLoss',
    'F',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'NAdam',
    'RAdam',
    'Adamax',
    'ASGD',
    'LBFGS',
    'get_optimizer',
    'LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    'WarmupScheduler',
    'CyclicLR',
    'OneCycleLR',
    
    # Data
    'Dataset',
    'TensorDataset',
    'ArrayDataset',
    'DataLoader',
    'SubsetDataset',
    'MappedDataset',
    'random_split',
    'Transform',
    'Compose',
    'ToTensor',
    'ToPILImage',
    'ToNumpy',
    'ToDtype',
    'Normalize',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomCrop',
    'CenterCrop',
    'Resize',
    'RandomAffine',
    'RandomPerspective',
    'ColorJitter',
    'RandomErasing',
    'RandomChoice',
    'RandomApply',
    'Lambda',
    
    # Metrics
    'Metric',
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'MSE',
    'MAE',
    'RMSE',
    'R2Score',
    'ConfusionMatrix',
    'AUC',
    'IoU',
    'DiceCoefficient',
    'Perplexity',
    'MetricCollection',
    'get_metric',
    
    # Utils
    'utils',
    
    # Version
    '__version__',
]

# Framework information
def info():
    """Print Texor framework information"""
    try:
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
    except ImportError:
        print(f"Texor v{__version__} - Native Deep Learning Framework")
        print(f"Device: {get_device()}")
        print(f"GPU Available: {cuda_is_available()}")

# Configure warnings
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)