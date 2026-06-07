import numpy as np
from typing import Optional, Union, Tuple
from ...core.native_tensor import Tensor
from .base import Layer

class Dropout(Layer):
    """Dropout layer"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.dropout(x, self.p, training=self.training)

class Dropout2D(Layer):
    """2D Dropout layer (drops entire channels)"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, (x.shape[0], x.shape[1], 1, 1)).astype(np.float32)
        return x * Tensor(mask) / (1 - self.p)

class Dropout3D(Layer):
    """3D Dropout layer (drops entire channels)"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, (x.shape[0], x.shape[1], 1, 1, 1)).astype(np.float32)
        return x * Tensor(mask) / (1 - self.p)

class Flatten(Layer):
    """Flatten layer"""
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.flatten(x, start_dim=1)

class Reshape(Layer):
    """Reshape layer"""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        shape = self.shape
        if shape[0] == -1:
            shape = (x.shape[0],) + shape[1:]
        return F.reshape(x, shape)

class PixelShuffle(Layer):
    """Pixel shuffle upsampling layer"""
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.pixel_shuffle(x, self.upscale_factor)

class Upsample(Layer):
    """Upsampling layer"""
    def __init__(self, size: Optional[Union[int, Tuple[int, ...]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
                 mode: str = 'nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.upsample(x, self.size, self.scale_factor, self.mode)

class LazyLinear(Layer):
    """Lazy linear layer that infers input features from first input"""
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.weight = None
        self.bias_tensor = None
        self._initialized = False

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialized:
            in_features = x.shape[-1]
            from .linear import Linear
            self._linear = Linear(in_features, self.out_features, self.bias)
            self._initialized = True
        return self._linear(x)
