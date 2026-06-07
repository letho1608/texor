from typing import Union, Tuple, Optional
import numpy as np
from ...core.native_tensor import Tensor
from ...core.native_backend import backend
from .base import Layer

class MaxPool2D(Layer):
    """2D Max Pooling layer"""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class AvgPool2D(Layer):
    """2D Average Pooling layer"""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

class MaxPool3D(Layer):
    """3D Max Pooling layer"""
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        # Implementation in functional would be better
        # For now placeholder or native loop
        N, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        
        D_out = (D + 2 * pD - kD) // sD + 1
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        
        result = np.zeros((N, C, D_out, H_out, W_out))
        # ... native loop ...
        return Tensor(result, requires_grad=x.requires_grad)

class AvgPool3D(Layer):
    """3D Average Pooling layer"""
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        # Implementation
        return x

class AdaptiveAvgPool2D(Layer):
    """2D Adaptive Average Pooling layer"""
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.adaptive_avg_pool2d(x, self.output_size)
