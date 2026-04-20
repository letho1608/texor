"""Functional interface to neural network operations

This module provides functional versions of neural network operations that can be
used without creating explicit layers. Inspired by PyTorch's torch.nn.functional.
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
from ..core import Tensor
from ..core.native_backend import backend


# =============================================================================
# Activation Functions
# =============================================================================

def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """Rectified Linear Unit activation function
    
    Args:
        x: Input tensor
        inplace: If True, modify input directly (not supported yet)
    
    Returns:
        Tensor after ReLU activation
    """
    if inplace:
        raise NotImplementedError("Inplace ReLU not yet supported")
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor after sigmoid activation
    """
    data = x.data
    positive_mask = data >= 0
    result = np.zeros_like(data)
    result[positive_mask] = 1 / (1 + np.exp(-data[positive_mask]))
    exp_x = np.exp(data[~positive_mask])
    result[~positive_mask] = exp_x / (1 + exp_x)
    return Tensor(result, requires_grad=x.requires_grad)


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent activation function
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor after tanh activation
    """
    return Tensor(np.tanh(x.data), requires_grad=x.requires_grad)


def leaky_relu(x: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    """Leaky ReLU activation function
    
    Args:
        x: Input tensor
        negative_slope: Slope for negative values
        inplace: If True, modify input directly (not supported yet)
    
    Returns:
        Tensor after LeakyReLU activation
    """
    if inplace:
        raise NotImplementedError("Inplace LeakyReLU not yet supported")
    return Tensor(np.where(x.data > 0, x.data, negative_slope * x.data),
                  requires_grad=x.requires_grad)


def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Exponential Linear Unit activation function
    
    Args:
        x: Input tensor
        alpha: Alpha value for ELU
        inplace: If True, modify input directly (not supported yet)
    
    Returns:
        Tensor after ELU activation
    """
    if inplace:
        raise NotImplementedError("Inplace ELU not yet supported")
    return Tensor(np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1)),
                  requires_grad=x.requires_grad)


def gelu(x: Tensor, approximate: str = 'none') -> Tensor:
    """Gaussian Error Linear Unit activation function
    
    Args:
        x: Input tensor
        approximate: If 'tanh', use tanh approximation (faster but less accurate)
    
    Returns:
        Tensor after GELU activation
    """
    if approximate == 'tanh':
        # Fast approximation
        return Tensor(0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3))),
                      requires_grad=x.requires_grad)
    else:
        # Exact GELU using error function
        from scipy.special import erf
        return Tensor(0.5 * x.data * (1 + erf(x.data / np.sqrt(2))),
                      requires_grad=x.requires_grad)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation function
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax
    
    Returns:
        Tensor after softmax activation
    """
    exp_x = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    return Tensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True),
                  requires_grad=x.requires_grad)


def softplus(x: Tensor) -> Tensor:
    """Softplus activation function
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor after softplus activation
    """
    return Tensor(np.log(1 + np.exp(x.data)), requires_grad=x.requires_grad)


def mish(x: Tensor) -> Tensor:
    """Mish activation function: x * tanh(softplus(x))
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor after Mish activation
    """
    return x * tanh(softplus(x))


# =============================================================================
# Dropout
# =============================================================================

def dropout(x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """Apply dropout to input tensor
    
    Args:
        x: Input tensor
        p: Probability of dropping elements
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modify input directly (not supported yet)
    
    Returns:
        Tensor after dropout
    """
    if not training or p == 0:
        return x
    
    if inplace:
        raise NotImplementedError("Inplace dropout not yet supported")
    
    mask = Tensor(
        np.random.binomial(1, 1 - p, x.shape).astype(np.float32)
    )
    return (x * mask) / (1 - p)


def alpha_dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Alpha dropout: a dropout that maintains the mean and variance of the inputs
    
    Args:
        x: Input tensor
        p: Probability of dropping elements
        training: If True, apply dropout
    
    Returns:
        Tensor after alpha dropout
    """
    if not training or p == 0:
        return x
    
    # Calculate alpha and shift values for alpha dropout
    alpha = -1.7580993408473766  # -1.7580993408473766
    a = (1 + p) ** 0.5 * (1 + p * alpha ** 2) ** 0.5
    b = -a * alpha
    
    mask = np.random.binomial(1, 1 - p, x.shape).astype(np.float32)
    return Tensor(a * (x.data * mask + alpha * (1 - mask)) + b,
                  requires_grad=x.requires_grad)


# =============================================================================
# Normalization
# =============================================================================

def layer_norm(x: Tensor, normalized_shape: Tuple[int, ...], 
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    """Layer normalization
    
    Args:
        x: Input tensor
        normalized_shape: Shape of the layer (last N dimensions)
        weight: Learnable scale parameter (gamma)
        bias: Learnable shift parameter (beta)
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    # Calculate the number of dimensions to normalize over
    ndims = len(normalized_shape)
    axis = tuple(range(-ndims, 0))
    
    mean = x.data.mean(axis=axis, keepdims=True)
    var = x.data.var(axis=axis, keepdims=True)
    x_norm = (x.data - mean) / np.sqrt(var + eps)
    
    result = x_norm
    if weight is not None:
        result = result * weight.data
    if bias is not None:
        result = result + bias.data
    
    return Tensor(result, requires_grad=x.requires_grad)


def batch_norm(x: Tensor, running_mean: Optional[Tensor], 
               running_var: Optional[Tensor], 
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               training: bool = False, momentum: float = 0.1, 
               eps: float = 1e-5) -> Tensor:
    """Batch normalization
    
    Args:
        x: Input tensor of shape (N, C, H, W) or (N, C)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        weight: Learnable scale parameter (gamma)
        bias: Learnable shift parameter (beta)
        training: If True, use batch statistics; if False, use running statistics
        momentum: Momentum for updating running statistics
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    if training:
        # Calculate batch statistics
        if x.data.ndim == 4:
            # (N, C, H, W) -> compute mean and var over N, H, W
            mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            var = x.data.var(axis=(0, 2, 3), keepdims=True)
        else:
            # (N, C) -> compute mean and var over N
            mean = x.data.mean(axis=0, keepdims=True)
            var = x.data.var(axis=0, keepdims=True)
        
        # Update running statistics
        if running_mean is not None and running_var is not None:
            mean_val = mean.squeeze() if mean.size == 1 else mean.squeeze(-1) if mean.ndim > 1 else mean
            var_val = var.squeeze() if var.size == 1 else var.squeeze(-1) if var.ndim > 1 else var
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean_val
            running_var.data = (1 - momentum) * running_var.data + momentum * var_val
    else:
        # Use running statistics
        if running_mean is None or running_var is None:
            raise ValueError("Running statistics required for inference mode")
        if x.data.ndim == 4:
            mean = running_mean.data.reshape(1, -1, 1, 1)
            var = running_var.data.reshape(1, -1, 1, 1)
        else:
            mean = running_mean.data.reshape(1, -1)
            var = running_var.data.reshape(1, -1)
    
    # Normalize
    x_norm = (x.data - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    result = x_norm
    if weight is not None:
        if x.data.ndim == 4:
            result = result * weight.data.reshape(1, -1, 1, 1)
        else:
            result = result * weight.data.reshape(1, -1)
    if bias is not None:
        if x.data.ndim == 4:
            result = result + bias.data.reshape(1, -1, 1, 1)
        else:
            result = result + bias.data.reshape(1, -1)
    
    return Tensor(result, requires_grad=x.requires_grad)


def instance_norm(x: Tensor, running_mean: Optional[Tensor] = None,
                 running_var: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                 use_input_stats: bool = True, momentum: float = 0.1,
                 eps: float = 1e-5) -> Tensor:
    """Instance normalization

    Args:
        x: Input tensor of shape (N, C, H, W) or (N, C)
        running_mean: Running mean (not used in instance norm)
        running_var: Running variance (not used in instance norm)
        weight: Learnable scale parameter
        bias: Learnable shift parameter
        use_input_stats: If True, use batch statistics
        momentum: Momentum (not used in instance norm)
        eps: Small value for numerical stability

    Returns:
        Normalized tensor
    """
    # Instance norm computes mean and var over spatial dimensions for each N, C
    if x.data.ndim == 4:
        mean = x.data.mean(axis=(2, 3), keepdims=True)
        var = x.data.var(axis=(2, 3), keepdims=True)
    else:
        # For 2D input (N, C), instance norm is identity
        mean = 0
        var = 1
    
    x_norm = (x.data - mean) / np.sqrt(var + eps)

    result = x_norm
    if weight is not None:
        if x.data.ndim == 4:
            result = result * weight.data.reshape(1, -1, 1, 1)
        else:
            result = result * weight.data.reshape(1, -1)
    if bias is not None:
        if x.data.ndim == 4:
            result = result + bias.data.reshape(1, -1, 1, 1)
        else:
            result = result + bias.data.reshape(1, -1)

    return Tensor(result, requires_grad=x.requires_grad)


def group_norm(x: Tensor, num_groups: int, weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    """Group normalization

    Args:
        x: Input tensor of shape (N, C, H, W) or (N, C)
        num_groups: Number of groups to split channels into
        weight: Learnable scale parameter
        bias: Learnable shift parameter
        eps: Small value for numerical stability

    Returns:
        Normalized tensor
    """
    if x.data.ndim == 4:
        N, C, H, W = x.data.shape
        assert C % num_groups == 0, "Number of channels must be divisible by num_groups"

        # Reshape to (N, num_groups, C // num_groups, H, W)
        x_reshaped = x.data.reshape(N, num_groups, C // num_groups, H, W)

        # Compute mean and var over the group dimensions (C // num_groups, H, W)
        mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)

        x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
        x_norm = x_norm.reshape(N, C, H, W)

        result = x_norm
        if weight is not None:
            result = result * weight.data.reshape(1, -1, 1, 1)
        if bias is not None:
            result = result + bias.data.reshape(1, -1, 1, 1)
    else:
        # For 2D input (N, C), group norm reduces to layer norm
        C = x.data.shape[-1]
        result = layer_norm(x, (C,), weight, bias, eps)

    return Tensor(result, requires_grad=x.requires_grad)


# =============================================================================
# Linear Operations
# =============================================================================

def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Apply linear transformation: y = xW^T + b
    
    Args:
        x: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features,)
    
    Returns:
        Output tensor of shape (*, out_features)
    """
    result = x @ weight.T
    if bias is not None:
        result = result + bias
    return result


def bilinear(x1: Tensor, x2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Apply bilinear transformation: y = x1^T W x2 + b
    
    Args:
        x1: First input tensor of shape (batch_size, in_features1)
        x2: Second input tensor of shape (batch_size, in_features2)
        weight: Weight tensor of shape (out_features, in_features1, in_features2)
        bias: Bias tensor of shape (out_features,)
    
    Returns:
        Output tensor
    """
    # Compute bilinear product
    result = np.einsum('bi,bj,bijk->bk', x1.data, x2.data, weight.data)
    if bias is not None:
        result = result + bias.data
    return Tensor(result, requires_grad=x1.requires_grad or x2.requires_grad)


# =============================================================================
# Convolution Operations
# =============================================================================

def conv1d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1) -> Tensor:
    """1D convolution
    
    Args:
        x: Input tensor of shape (N, C_in, L_in)
        weight: Weight tensor of shape (C_out, C_in, k)
        bias: Bias tensor of shape (C_out,)
        stride: Stride value
        padding: Padding value
        dilation: Dilation value
    
    Returns:
        Output tensor of shape (N, C_out, L_out)
    """
    # Simple 1D convolution implementation
    if padding > 0:
        x = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    
    N, C_in, L_in = x.data.shape
    C_out, _, k = weight.data.shape
    
    L_out = (L_in - dilation * (k - 1) - 1) // stride + 1
    
    result = np.zeros((N, C_out, L_out))
    
    for i in range(L_out):
        start = i * stride
        end = start + dilation * (k - 1) + 1
        if end > L_in:
            break
        # Extract patches with dilation
        patch = x.data[:, :, start:end:dilation]  # N, C_in, k
        for c_out in range(C_out):
            result[:, c_out, i] = np.sum(patch * weight.data[c_out], axis=(1, 2))
    
    output = Tensor(result, requires_grad=x.requires_grad)

    if bias is not None:
        output = output + bias.data.reshape(1, -1, 1)

    return output


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[int, Tuple[int, int]] = 0,
           dilation: Union[int, Tuple[int, int]] = 1) -> Tensor:
    """2D convolution

    Args:
        x: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_out, C_in, kH, kW)
        bias: Bias tensor of shape (C_out,)
        stride: Stride value (single int or tuple)
        padding: Padding value (single int or tuple)
        dilation: Dilation value (single int or tuple)

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    # Backend only supports stride and padding, not dilation
    return Tensor(backend.conv2d(
        x.data, weight.data,
        stride=stride_h,
        padding=pad_h
    ), requires_grad=x.requires_grad)


def conv3d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, int, int]] = 1,
           padding: Union[int, Tuple[int, int, int]] = 0,
           dilation: Union[int, Tuple[int, int, int]] = 1) -> Tensor:
    """3D convolution
    
    Args:
        x: Input tensor of shape (N, C_in, D_in, H_in, W_in)
        weight: Weight tensor of shape (C_out, C_in, kD, kH, kW)
        bias: Bias tensor of shape (C_out,)
        stride: Stride value
        padding: Padding value
        dilation: Dilation value
    
    Returns:
        Output tensor of shape (N, C_out, D_out, H_out, W_out)
    """
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    # 3D convolution implementation
    N, C_in, D_in, H_in, W_in = x.data.shape
    C_out, _, kD, kH, kW = weight.data.shape
    D_out = (D_in + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0] + 1
    H_out = (H_in + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1] + 1
    W_out = (W_in + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2] + 1
    
    result = np.zeros((N, C_out, D_out, H_out, W_out))
    
    # Apply padding
    if padding != (0, 0, 0):
        x_padded = np.pad(x.data, 
                         ((0, 0), (0, 0), 
                          (padding[0], padding[0]),
                          (padding[1], padding[1]),
                          (padding[2], padding[2])), 
                         mode='constant')
    else:
        x_padded = x.data
    
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                d_start = d * stride[0]
                h_start = h * stride[1]
                w_start = w * stride[2]
                
                patch = x_padded[:, :,
                                d_start:d_start + dilation[0] * kD:dilation[0],
                                h_start:h_start + dilation[1] * kH:dilation[1],
                                w_start:w_start + dilation[2] * kW:dilation[2]]
                
                for c_out in range(C_out):
                    result[:, c_out, d, h, w] = np.sum(
                        patch * weight.data[c_out], axis=(1, 2, 3, 4)
                    )
    
    output = Tensor(result, requires_grad=x.requires_grad)

    if bias is not None:
        output = output + bias.data.reshape(1, -1, 1, 1, 1)

    return output


def conv_transpose2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                     stride: Union[int, Tuple[int, int]] = 1,
                     padding: Union[int, Tuple[int, int]] = 0,
                     output_padding: int = 0,
                     dilation: Union[int, Tuple[int, int]] = 1) -> Tensor:
    """2D transposed convolution (deconvolution)
    
    Args:
        x: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, kH, kW)
        bias: Bias tensor of shape (C_out,)
        stride: Stride value
        padding: Padding value
        output_padding: Additional size added to output
        dilation: Dilation value
    
    Returns:
        Output tensor
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    N, C_in, H_in, W_in = x.data.shape
    C_out, _, kH, kW = weight.data.shape
    
    # Calculate output size
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + output_padding + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + output_padding + 1
    
    # For simplicity, use native backend if available
    result = np.zeros((N, C_out, H_out, W_out))
    
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for h in range(H_in):
                    for w in range(W_in):
                        h_start = h * stride[0] - padding[0]
                        w_start = w * stride[1] - padding[1]
                        
                        for kh in range(kH):
                            for kw in range(kW):
                                hh = h_start + kh * dilation[0]
                                ww = w_start + kw * dilation[1]
                                if 0 <= hh < H_out and 0 <= ww < W_out:
                                    result[n, c_out, hh, ww] += (
                                        x.data[n, c_in, h, w] * weight.data[c_in, c_out, kh, kw]
                                    )
    
    output = Tensor(result, requires_grad=x.requires_grad)

    if bias is not None:
        output = output + bias.data.reshape(1, -1, 1, 1)

    return output


# =============================================================================
# Pooling Operations
# =============================================================================

def avg_pool1d(x: Tensor, kernel_size: int, stride: Optional[int] = None,
               padding: int = 0) -> Tensor:
    """1D average pooling
    
    Args:
        x: Input tensor of shape (N, C, L_in)
        kernel_size: Size of the pooling window
        stride: Stride value (defaults to kernel_size)
        padding: Padding value
    
    Returns:
        Output tensor of shape (N, C, L_out)
    """
    if stride is None:
        stride = kernel_size
    
    N, C, L_in = x.data.shape
    
    if padding > 0:
        x = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        L_in += 2 * padding
    
    L_out = (L_in - kernel_size) // stride + 1
    
    result = np.zeros((N, C, L_out))
    
    for i in range(L_out):
        start = i * stride
        end = start + kernel_size
        result[:, :, i] = x.data[:, :, start:end].mean(axis=2)
    
    return Tensor(result, requires_grad=x.requires_grad)


def avg_pool2d(x: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> Tensor:
    """2D average pooling
    
    Args:
        x: Input tensor of shape (N, C, H_in, W_in)
        kernel_size: Size of the pooling window
        stride: Stride value (defaults to kernel_size)
        padding: Padding value
    
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    N, C, H_in, W_in = x.data.shape
    
    if padding != (0, 0):
        x_padded = np.pad(x.data, 
                         ((0, 0), (0, 0), 
                          (padding[0], padding[0]),
                          (padding[1], padding[1])), 
                         mode='constant')
    else:
        x_padded = x.data
    
    H_in_padded = H_in + 2 * padding[0]
    W_in_padded = W_in + 2 * padding[1]
    
    H_out = (H_in_padded - kernel_size[0]) // stride[0] + 1
    W_out = (W_in_padded - kernel_size[1]) // stride[1] + 1
    
    result = np.zeros((N, C, H_out, W_out))
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride[0]
            w_start = w * stride[1]
            patch = x_padded[:, :, 
                           h_start:h_start + kernel_size[0],
                           w_start:w_start + kernel_size[1]]
            result[:, :, h, w] = patch.mean(axis=(2, 3))
    
    return Tensor(result, requires_grad=x.requires_grad)


def max_pool2d(x: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> Tensor:
    """2D max pooling
    
    Args:
        x: Input tensor of shape (N, C, H_in, W_in)
        kernel_size: Size of the pooling window
        stride: Stride value (defaults to kernel_size)
        padding: Padding value
    
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    N, C, H_in, W_in = x.data.shape
    
    if padding != (0, 0):
        x_padded = np.pad(x.data, 
                         ((0, 0), (0, 0), 
                          (padding[0], padding[0]),
                          (padding[1], padding[1])), 
                         mode='constant',
                         constant_values=float('-inf'))
    else:
        x_padded = x.data
    
    H_in_padded = H_in + 2 * padding[0]
    W_in_padded = W_in + 2 * padding[1]
    
    H_out = (H_in_padded - kernel_size[0]) // stride[0] + 1
    W_out = (W_in_padded - kernel_size[1]) // stride[1] + 1
    
    result = np.zeros((N, C, H_out, W_out))
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride[0]
            w_start = w * stride[1]
            patch = x_padded[:, :, 
                           h_start:h_start + kernel_size[0],
                           w_start:w_start + kernel_size[1]]
            result[:, :, h, w] = patch.max(axis=(2, 3))
    
    return Tensor(result, requires_grad=x.requires_grad)


def adaptive_avg_pool2d(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """2D adaptive average pooling
    
    Args:
        x: Input tensor of shape (N, C, H_in, W_in)
        output_size: Output size (single int or tuple)
    
    Returns:
        Output tensor of shape (N, C, output_size[0], output_size[1])
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    N, C, H_in, W_in = x.data.shape
    H_out, W_out = output_size
    
    # Calculate kernel size and stride
    kh = H_in // H_out
    kw = W_in // W_out
    
    result = np.zeros((N, C, H_out, W_out))
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * kh
            w_start = w * kw
            # Handle case where H_in or W_in is not divisible by output_size
            h_end = min(h_start + kh if h < H_out - 1 or H_in % H_out == 0 else H_in, H_in)
            w_end = min(w_start + kw if w < W_out - 1 or W_in % W_out == 0 else W_in, W_in)
            result[:, :, h, w] = x.data[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))
    
    return Tensor(result, requires_grad=x.requires_grad)


def adaptive_max_pool2d(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """2D adaptive max pooling
    
    Args:
        x: Input tensor of shape (N, C, H_in, W_in)
        output_size: Output size (single int or tuple)
    
    Returns:
        Output tensor of shape (N, C, output_size[0], output_size[1])
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    N, C, H_in, W_in = x.data.shape
    H_out, W_out = output_size
    
    # Calculate kernel size and stride
    kh = H_in // H_out
    kw = W_in // W_out
    
    result = np.zeros((N, C, H_out, W_out))
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * kh
            w_start = w * kw
            h_end = min(h_start + kh if h < H_out - 1 or H_in % H_out == 0 else H_in, H_in)
            w_end = min(w_start + kw if w < W_out - 1 or W_in % W_out == 0 else W_in, W_in)
            result[:, :, h, w] = x.data[:, :, h_start:h_end, w_start:w_end].max(axis=(2, 3))
    
    return Tensor(result, requires_grad=x.requires_grad)


# =============================================================================
# Normalization Functions
# =============================================================================

def normalize(x: Tensor, mean: np.ndarray, std: np.ndarray, 
              inplace: bool = False) -> Tensor:
    """Normalize a tensor with mean and std
    
    Args:
        x: Input tensor
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        inplace: If True, modify input directly
    
    Returns:
        Normalized tensor
    """
    mean = np.array(mean).reshape(1, -1, 1, 1) if x.data.ndim == 4 else np.array(mean).reshape(1, -1)
    std = np.array(std).reshape(1, -1, 1, 1) if x.data.ndim == 4 else np.array(std).reshape(1, -1)
    
    if inplace:
        x.data = (x.data - mean) / std
        return x
    
    return Tensor((x.data - mean) / std, requires_grad=x.requires_grad)


# =============================================================================
# Utility Functions
# =============================================================================

def pad(x: Tensor, pad: Tuple[int, ...], mode: str = 'constant', 
        value: float = 0) -> Tensor:
    """Pad tensor
    
    Args:
        x: Input tensor
        pad: Padding sizes (must be in format: (left, right, top, bottom, ...) for 4D)
        mode: Padding mode ('constant', 'reflect', 'replicate')
        value: Fill value for constant mode
    
    Returns:
        Padded tensor
    """
    return Tensor(np.pad(x.data, pad, mode=mode, constant_values=value),
                  requires_grad=x.requires_grad)


def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten tensor
    
    Args:
        x: Input tensor
        start_dim: First dimension to flatten
        end_dim: Last dimension to flatten
    
    Returns:
        Flattened tensor
    """
    if end_dim == -1:
        end_dim = x.data.ndim - 1
    
    # Calculate new shape
    shape = list(x.shape)
    flatten_size = np.prod(shape[start_dim:end_dim + 1])
    new_shape = shape[:start_dim] + [flatten_size] + shape[end_dim + 1:]
    
    return Tensor(x.data.reshape(new_shape), requires_grad=x.requires_grad)


def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape tensor
    
    Args:
        x: Input tensor
        shape: New shape
    
    Returns:
        Reshaped tensor
    """
    return Tensor(x.data.reshape(shape), requires_grad=x.requires_grad)


def transpose(x: Tensor, axes: Tuple[int, ...]) -> Tensor:
    """Transpose tensor
    
    Args:
        x: Input tensor
        axes: Tuple of axes to transpose
    
    Returns:
        Transposed tensor
    """
    return Tensor(np.transpose(x.data, axes), requires_grad=x.requires_grad)


def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1
    
    Args:
        x: Input tensor
        dim: Dimension to remove (if None, remove all size-1 dimensions)
    
    Returns:
        Squeezed tensor
    """
    if dim is None:
        return Tensor(np.squeeze(x.data), requires_grad=x.requires_grad)
    else:
        return Tensor(np.squeeze(x.data, axis=dim), requires_grad=x.requires_grad)


def unsqueeze(x: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1
    
    Args:
        x: Input tensor
        dim: Position to add dimension
    
    Returns:
        Unsqueezed tensor
    """
    return Tensor(np.expand_dims(x.data, dim), requires_grad=x.requires_grad)


def cat(tensors: Tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Concatenate tensors along dimension
    
    Args:
        tensors: Tuple of tensors to concatenate
        dim: Dimension along which to concatenate
    
    Returns:
        Concatenated tensor
    """
    return Tensor(np.concatenate([t.data for t in tensors], axis=dim),
                  requires_grad=any(t.requires_grad for t in tensors))


def stack(tensors: Tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Stack tensors along new dimension
    
    Args:
        tensors: Tuple of tensors to stack
        dim: Dimension along which to stack
    
    Returns:
        Stacked tensor
    """
    return Tensor(np.stack([t.data for t in tensors], axis=dim),
                  requires_grad=any(t.requires_grad for t in tensors))


def split(x: Tensor, split_size_or_sections: Union[int, Tuple[int, ...]],
          dim: int = 0) -> Tuple[Tensor, ...]:
    """Split tensor into chunks
    
    Args:
        x: Input tensor
        split_size_or_sections: Size of each chunk or list of sizes
        dim: Dimension along which to split
    
    Returns:
        Tuple of tensors
    """
    if isinstance(split_size_or_sections, int):
        splits = np.split(x.data, split_size_or_sections, axis=dim)
    else:
        splits = np.split(x.data, np.cumsum(split_size_or_sections[:-1]), axis=dim)
    
    return tuple(Tensor(s, requires_grad=x.requires_grad) for s in splits)


def chunk(x: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """Split tensor into chunks
    
    Args:
        x: Input tensor
        chunks: Number of chunks
        dim: Dimension along which to split
    
    Returns:
        Tuple of tensors
    """
    return split(x, x.shape[dim] // chunks, dim=dim)


def repeat(x: Tensor, *repeats) -> Tensor:
    """Repeat tensor elements
    
    Args:
        x: Input tensor
        repeats: Number of repeats for each dimension
    
    Returns:
        Repeated tensor
    """
    return Tensor(np.repeat(x.data, *repeats), requires_grad=x.requires_grad)


def tile(x: Tensor, reps: Tuple[int, ...]) -> Tensor:
    """Tile tensor
    
    Args:
        x: Input tensor
        reps: Number of repetitions for each dimension
    
    Returns:
        Tiled tensor
    """
    return Tensor(np.tile(x.data, reps), requires_grad=x.requires_grad)


def gather(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values from tensor
    
    Args:
        x: Input tensor
        dim: Dimension along which to gather
        index: Indices to gather
    
    Returns:
        Gathered tensor
    """
    return Tensor(np.take_along_axis(x.data, index.data, axis=dim),
                  requires_grad=x.requires_grad)


def scatter_add(x: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Scatter add values to tensor
    
    Args:
        x: Input tensor
        dim: Dimension along which to scatter
        index: Indices to scatter to
        src: Values to scatter
    
    Returns:
        Updated tensor
    """
    result = x.data.copy()
    np.add.at(result, (slice(None),) * dim + (index.data,), src.data)
    return Tensor(result, requires_grad=x.requires_grad)


# =============================================================================
# Loss Functions (Functional Form)
# =============================================================================

def binary_cross_entropy_with_logits(x: Tensor, target: Tensor,
                                     weight: Optional[Tensor] = None,
                                     reduction: str = 'mean') -> Tensor:
    """Binary cross entropy with logits (combines sigmoid and BCE)
    
    Args:
        x: Input tensor (raw logits)
        target: Target tensor (0 or 1)
        weight: Optional weight tensor
        reduction: Reduction method ('none', 'mean', 'sum')
    
    Returns:
        Loss tensor
    """
    # Sigmoid + BCE combined for numerical stability
    x = sigmoid(x)
    x = np.clip(x.data, 1e-7, 1 - 1e-7)
    losses = -(target.data * np.log(x) + (1 - target.data) * np.log(1 - x))
    
    if weight is not None:
        losses = losses * weight.data
    
    if reduction == 'none':
        return Tensor(losses, requires_grad=x.requires_grad)
    elif reduction == 'mean':
        return Tensor(np.mean(losses), requires_grad=x.requires_grad)
    else:
        return Tensor(np.sum(losses), requires_grad=x.requires_grad)


def nll_loss(x: Tensor, target: Tensor, weight: Optional[Tensor] = None,
             ignore_index: int = -100, reduction: str = 'mean') -> Tensor:
    """Negative log likelihood loss
    
    Args:
        x: Input tensor (log-probabilities)
        target: Target tensor (class indices)
        weight: Optional weight tensor
        ignore_index: Index to ignore
        reduction: Reduction method
    
    Returns:
        Loss tensor
    """
    # Get log probabilities
    log_prob = x.data
    
    # Handle class indices
    if target.data.ndim == 1:
        # Convert to negative log likelihood
        batch_indices = np.arange(len(target.data))
        class_indices = target.data.astype(int)
        losses = -log_prob[batch_indices, class_indices]
    else:
        losses = -np.sum(target.data * log_prob, axis=-1)
    
    # Handle ignore_index
    mask = target.data != ignore_index
    losses = losses * mask
    
    if weight is not None:
        losses = losses * weight.data[target.data.astype(int)]
    
    if reduction == 'none':
        return Tensor(losses, requires_grad=x.requires_grad)
    elif reduction == 'mean':
        return Tensor(np.mean(losses[mask]), requires_grad=x.requires_grad)
    else:
        return Tensor(np.sum(losses[mask]), requires_grad=x.requires_grad)


# =============================================================================
# Distance Functions
# =============================================================================

def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    """Compute pairwise distances between two sets of vectors
    
    Args:
        x1: First tensor of shape (N, D)
        x2: Second tensor of shape (M, D)
        p: Power parameter for Minkowski distance
    
    Returns:
        Distance tensor of shape (N, M)
    """
    # Use broadcasting to compute distances
    x1_expanded = x1.data[:, np.newaxis, :]
    x2_expanded = x2.data[np.newaxis, :, :]
    
    if p == 1.0:
        distances = np.abs(x1_expanded - x2_expanded).sum(axis=2)
    elif p == 2.0:
        distances = np.sqrt(((x1_expanded - x2_expanded) ** 2).sum(axis=2))
    else:
        distances = ((np.abs(x1_expanded - x2_expanded)) ** p).sum(axis=2) ** (1/p)
    
    return Tensor(distances, requires_grad=x1.requires_grad or x2.requires_grad)


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Compute cosine similarity between vectors
    
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small value for numerical stability
    
    Returns:
        Similarity tensor
    """
    x1_norm = np.linalg.norm(x1.data, axis=dim, keepdims=True)
    x2_norm = np.linalg.norm(x2.data, axis=dim, keepdims=True)
    
    similarity = (x1 * x2).sum(dim=dim) / (x1_norm * x2_norm + eps)
    
    return similarity


def pdist(x: Tensor, p: float = 2.0) -> Tensor:
    """Compute pairwise distances between rows of x
    
    Args:
        x: Input tensor of shape (N, D)
        p: Power parameter for Minkowski distance
    
    Returns:
        Distance tensor of shape (N*(N-1)/2,)
    """
    N = x.data.shape[0]
    
    # Compute all pairwise distances
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = x.data[i] - x.data[j]
            if p == 1.0:
                result.append(np.abs(diff).sum())
            elif p == 2.0:
                result.append(np.sqrt((diff ** 2).sum()))
            else:
                result.append((np.abs(diff) ** p).sum() ** (1/p))
    
    return Tensor(np.array(result), requires_grad=x.requires_grad)


def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    """Compute distances between rows of x1 and x2
    
    Args:
        x1: First tensor of shape (N, D)
        x2: Second tensor of shape (M, D)
        p: Power parameter for Minkowski distance
    
    Returns:
        Distance tensor of shape (N, M)
    """
    return pairwise_distance(x1, x2, p)


# =============================================================================
# Grid Sampling
# =============================================================================

def grid_sample(x: Tensor, grid: Tensor, mode: str = 'bilinear',
                padding_mode: str = 'zeros', align_corners: bool = False) -> Tensor:
    """Sample input tensor using coordinates from grid
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        grid: Grid tensor of shape (N, H_out, W_out, 2) with coordinates in [-1, 1]
        mode: Interpolation mode ('bilinear' or 'nearest')
        padding_mode: Padding mode for out-of-bound coordinates
        align_corners: If True, corner pixels are aligned
    
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.data.shape
    H_out, W_out, _ = grid.data.shape
    
    # Convert grid from [-1, 1] to pixel coordinates
    if align_corners:
        x_grid = (grid.data[:, :, :, 0] + 1) * (W - 1) / 2
        y_grid = (grid.data[:, :, :, 1] + 1) * (H - 1) / 2
    else:
        x_grid = ((grid.data[:, :, :, 0] + 1) * W - 1) / 2
        y_grid = ((grid.data[:, :, :, 1] + 1) * H - 1) / 2
    
    result = np.zeros((N, C, H_out, W_out))
    
    if mode == 'bilinear':
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        x = x_grid[n, h, w]
                        y = y_grid[n, h, w]
                        
                        # Handle padding
                        if padding_mode == 'zeros':
                            if x < 0 or x > W - 1 or y < 0 or y > H - 1:
                                continue
                        elif padding_mode == 'border':
                            x = max(0, min(W - 1, x))
                            y = max(0, min(H - 1, y))
                        
                        # Bilinear interpolation
                        x0 = int(np.floor(x))
                        y0 = int(np.floor(y))
                        x1 = x0 + 1
                        y1 = y0 + 1
                        
                        if x1 < W and y1 < H:
                            wx = x - x0
                            wy = y - y0
                            
                            v00 = x.data[n, c, y0, x0]
                            v01 = x.data[n, c, y0, x1] if x1 < W else v00
                            v10 = x.data[n, c, y1, x0] if y1 < H else v00
                            v11 = x.data[n, c, y1, x1] if y1 < H and x1 < W else v00
                            
                            result[n, c, h, w] = (
                                v00 * (1 - wx) * (1 - wy) +
                                v01 * wx * (1 - wy) +
                                v10 * (1 - wx) * wy +
                                v11 * wx * wy
                            )
    else:  # nearest
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        x = int(np.round(x_grid[n, h, w]))
                        y = int(np.round(y_grid[n, h, w]))
                        
                        if padding_mode == 'zeros':
                            if 0 <= x < W and 0 <= y < H:
                                result[n, c, h, w] = x.data[n, c, y, x]
                        else:
                            x = max(0, min(W - 1, x))
                            y = max(0, min(H - 1, y))
                            result[n, c, h, w] = x.data[n, c, y, x]
    
    return Tensor(result, requires_grad=x.requires_grad)


def affine_grid(theta: Tensor, size: Tuple[int, int, int, int],
                align_corners: bool = False) -> Tensor:
    """Generate affine grid for sampling
    
    Args:
        theta: Transformation matrix of shape (N, 2, 3)
        size: Output size (N, C, H, W)
        align_corners: If True, corner pixels are aligned
    
    Returns:
        Grid tensor of shape (N, H, W, 2)
    """
    N, _, _, _ = size
    _, _, H, W = size
    
    # Create base grid
    if align_corners:
        x_grid = np.linspace(-1, 1, W)
        y_grid = np.linspace(-1, 1, H)
    else:
        x_grid = np.linspace(-1 + 1/W, 1 - 1/W, W)
        y_grid = np.linspace(-1 + 1/H, 1 - 1/H, H)
    
    y_grid, x_grid = np.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Stack base grid
    base_grid = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)  # (H, W, 3)
    
    # Apply transformation
    theta_data = theta.data.reshape(N, 2, 3)
    grids = np.einsum('hwc,ijc->ijhw', base_grid, theta_data[:, :, :2])
    grids = grids + theta_data[:, :, 2:].reshape(N, 1, 1, 2)
    
    return Tensor(grids, requires_grad=theta.requires_grad)