"""Neural network module for Texor"""

# Import functional module
from . import functional as F

# Import layers
from .layers import (
    Layer,
    Linear,
    Conv2D,
    MaxPool2D,
    BatchNorm2D,
    Sequential
)

# Import activations
from .activations import (
    ReLU,
    Sigmoid
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
    get_loss_function
)

# Import model
from .model import Model

# Define what's available when using "from texor.nn import *"
__all__ = [
    'F',
    'Layer', 'Linear', 'Conv2D', 'MaxPool2D', 'BatchNorm2D', 'Sequential',
    'ReLU', 'Sigmoid',
    'MSELoss', 'CrossEntropyLoss', 'BCELoss', 'L1Loss', 'HuberLoss', 
    'SmoothL1Loss', 'KLDivLoss', 'get_loss_function',
    'Model'
]
