"""Neural network module for Nexor"""
from . import functional as F

from .layers import (
    Layer,
    Linear,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    BatchNorm2d,
    AdaptiveAvgPool2d,
    Dropout,
    Sequential,
    Reshape,
    Flatten,
    Embedding,
    LayerNorm
)

from .activations import (
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Softmax,
    GELU
)

from .loss import (
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    L1Loss,
    HuberLoss
)

from .model import Model

from .models import (
    # ResNet models
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    
    # Transformer models
    TransformerEncoder,
    BERT,
    bert_base_uncased,
    bert_large_uncased,
    
    # GAN models
    GAN,
    DCGAN,
    create_dcgan
)

__all__ = [
    # Functional interface
    'F',
    
    # Base classes
    'Layer',
    'Model',
    
    # Basic layers
    'Linear',
    'Conv2d',
    'ConvTranspose2d',
    'MaxPool2d',
    'BatchNorm2d',
    'AdaptiveAvgPool2d',
    'Dropout',
    'Sequential',
    'Reshape',
    'Flatten',
    'Embedding',
    'LayerNorm',
    
    # Activation functions
    'ReLU',
    'Sigmoid', 
    'Tanh',
    'LeakyReLU',
    'ELU',
    'Softmax',
    'GELU',
    
    # Loss functions
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    'HuberLoss',
    
    # Pre-built models
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'TransformerEncoder',
    'BERT',
    'bert_base_uncased',
    'bert_large_uncased',
    'GAN',
    'DCGAN',
    'create_dcgan'
]