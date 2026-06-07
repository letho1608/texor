from typing import Union, Tuple, Optional
import numpy as np
from ...core.native_tensor import Tensor
from .base import Layer

class BatchNorm2D(Layer):
    """2D Batch Normalization layer"""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        from .. import init
        self.weight = Tensor(np.empty(num_features), requires_grad=True)
        init.ones(self.weight)
        self.bias = Tensor(np.empty(num_features), requires_grad=True)
        init.zeros(self.bias)
        
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
        self.running_var = Tensor(np.ones(num_features), requires_grad=False)
        
    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            batch_mean = inputs.mean(axis=(0, 2, 3))
            batch_var = inputs.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        x_norm = (inputs - batch_mean.reshape(1, -1, 1, 1)) / np.sqrt(batch_var.reshape(1, -1, 1, 1) + self.eps)
        return self.weight.reshape(1, -1, 1, 1) * x_norm + self.bias.reshape(1, -1, 1, 1)

class LayerNorm(Layer):
    """Layer Normalization"""
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        
        from .. import init
        self.weight = Tensor(np.empty(normalized_shape), requires_grad=True)
        init.ones(self.weight)
        self.bias = Tensor(np.empty(normalized_shape), requires_grad=True)
        init.zeros(self.bias)
        
    def forward(self, inputs: Tensor) -> Tensor:
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = inputs.mean(axis=axes, keepdims=True)
        var = inputs.var(axis=axes, keepdims=True)
        x_norm = (inputs - mean) / np.sqrt(var + self.eps)
        shape = [1] * (inputs.dim() - len(self.normalized_shape)) + list(self.normalized_shape)
        return self.weight.reshape(shape) * x_norm + self.bias.reshape(shape)

class GroupNorm(Layer):
    """Group Normalization"""
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        from .. import init
        self.weight = Tensor(np.empty(num_channels), requires_grad=True)
        init.ones(self.weight)
        self.bias = Tensor(np.empty(num_channels), requires_grad=True)
        init.zeros(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        N, C, H, W = inputs.shape
        assert C % self.num_groups == 0
        x = inputs.reshape(N, self.num_groups, C // self.num_groups, H, W)
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = x.var(axis=(2, 3, 4), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, H, W)
        return self.weight.reshape(1, -1, 1, 1) * x_norm + self.bias.reshape(1, -1, 1, 1)

class InstanceNorm1D(Layer):
    """1D Instance Normalization"""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        from .. import init
        self.weight = Tensor(np.empty(num_features), requires_grad=True)
        init.ones(self.weight)
        self.bias = Tensor(np.empty(num_features), requires_grad=True)
        init.zeros(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(axis=2, keepdims=True)
        var = inputs.var(axis=2, keepdims=True)
        x_norm = (inputs - mean) / np.sqrt(var + self.eps)
        return self.weight.reshape(1, -1, 1) * x_norm + self.bias.reshape(1, -1, 1)

class InstanceNorm2D(Layer):
    """2D Instance Normalization"""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        from .. import init
        self.weight = Tensor(np.empty(num_features), requires_grad=True)
        init.ones(self.weight)
        self.bias = Tensor(np.empty(num_features), requires_grad=True)
        init.zeros(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(axis=(2, 3), keepdims=True)
        var = inputs.var(axis=(2, 3), keepdims=True)
        x_norm = (inputs - mean) / np.sqrt(var + self.eps)
        return self.weight.reshape(1, -1, 1, 1) * x_norm + self.bias.reshape(1, -1, 1, 1)
