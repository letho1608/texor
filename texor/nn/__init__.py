"""Neural network module for Texor

This module provides all neural network components including layers,
activations, loss functions, and functional operations.
"""

# Import functional module
from . import functional as F

# Import layers from the new layers subpackage
from .layers import (
    Layer,
    Linear,
    Embedding,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose2D,
    MaxPool2D,
    AvgPool2D,
    MaxPool3D,
    AvgPool3D,
    AdaptiveAvgPool2D,
    BatchNorm2D,
    LayerNorm,
    GroupNorm,
    InstanceNorm1D,
    InstanceNorm2D,
    Sequential,
    ModuleList,
    Dropout,
    Dropout2D,
    Dropout3D,
    Flatten,
    Reshape,
    PixelShuffle,
    Upsample,
    LazyLinear,
    RNN,
    LSTM,
    GRU,
    SelfAttention,
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer
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

# Import init
from . import init

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
    'Embedding',
    'Conv1D',
    'Conv2D',
    'Conv3D',
    'ConvTranspose2D',
    'MaxPool2D',
    'AvgPool2D',
    'MaxPool3D',
    'AvgPool3D',
    'AdaptiveAvgPool2D',
    'BatchNorm2D',
    'LayerNorm',
    'GroupNorm',
    'InstanceNorm1D',
    'InstanceNorm2D',
    'Sequential',
    'ModuleList',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'Flatten',
    'Reshape',
    'PixelShuffle',
    'Upsample',
    'LazyLinear',
    'RNN',
    'LSTM',
    'GRU',
    'SelfAttention',
    'MultiheadAttention',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
    
    # Activations
    'ReLU',
    'Sigmoid',
    'Tanh',
    'LeakyReLU',
    'ELU',
    'Softmax',
    'GELU',
    'get_activation',
    
    # Init
    'init',
    
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
