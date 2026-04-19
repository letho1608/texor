"""Neural network module for Texor

This module provides all neural network components including layers,
activations, loss functions, and functional operations.
"""

# Import functional module
from . import functional as F

# Import layers
from .layers import (
    Layer,
    Linear,
    Conv2D,
    Conv1D,
    Conv3D,
    ConvTranspose2d,
    MaxPool2D,
    MaxPool3D,
    AvgPool3D,
    BatchNorm2D,
    LayerNorm,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    Dropout,
    Dropout2D,
    Dropout3D,
    AdaptiveAvgPool2d,
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
    PixelShuffle,
    Upsample,
    LazyLinear,
    Reshape as ReshapeLayer,
)

# Import activations
from .activations import (
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Softmax,
    GELU,
    get_activation,
)

# Import loss functions
from .loss import (
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    L1Loss,
    HuberLoss,
    SmoothL1Loss,
    KLDivLoss,
    get_loss_function,
)

# Import model
from .model import Model

# Define what's available when using "from texor.nn import *"
__all__ = [
    # Functional module
    'F',
    
    # Layers
    'Layer',
    'Linear',
    'Conv2D',
    'Conv1D',
    'Conv3D',
    'ConvTranspose2d',
    'MaxPool2D',
    'MaxPool3D',
    'AvgPool3D',
    'BatchNorm2D',
    'LayerNorm',
    'GroupNorm',
    'InstanceNorm1d',
    'InstanceNorm2d',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'AdaptiveAvgPool2d',
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
    'PixelShuffle',
    'Upsample',
    'LazyLinear',
    
    # Activations
    'ReLU',
    'Sigmoid',
    'Tanh',
    'LeakyReLU',
    'ELU',
    'Softmax',
    'GELU',
    'get_activation',
    
    # Loss functions
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    'HuberLoss',
    'SmoothL1Loss',
    'KLDivLoss',
    'get_loss_function',
    
    # Model
    'Model',
]
