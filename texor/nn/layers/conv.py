from typing import Union, Tuple, Optional
import numpy as np
from ...core.native_tensor import Tensor
from ...core.native_backend import backend
from .base import Layer

class Conv1D(Layer):
    """1D Convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        from .. import init
        self.weight = Tensor(np.empty((out_channels, in_channels, kernel_size)), requires_grad=True)
        init.kaiming_uniform(self.weight, nonlinearity='relu')
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding)

class Conv2D(Layer):
    """2D Convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        from .. import init
        self.weight = Tensor(np.empty((out_channels, in_channels, *self.kernel_size)), requires_grad=True)
        init.kaiming_uniform(self.weight, nonlinearity='relu')
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None
            
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

class Conv3D(Layer):
    """3D Convolution layer"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        from .. import init
        self.weight = Tensor(np.empty((out_channels, in_channels, *self.kernel_size)), requires_grad=True)
        init.kaiming_uniform(self.weight, nonlinearity='relu')
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding)

class ConvTranspose2D(Layer):
    """2D Transposed Convolution layer"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        from .. import init
        self.weight = Tensor(np.empty((in_channels, out_channels, *self.kernel_size)), requires_grad=True)
        init.kaiming_uniform(self.weight, nonlinearity='relu')
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None
        
    def forward(self, x: Tensor) -> Tensor:
        from .. import functional as F
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)
