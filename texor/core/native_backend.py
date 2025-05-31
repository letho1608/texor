"""
Native backend implementation for Texor
Independent from TensorFlow and PyTorch but incorporates their best practices
Performance optimized with JIT compilation and GPU support
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from numba import jit, prange, config
import warnings
import threading
import time

# Configure Numba for optimal performance
config.THREADING_LAYER = 'threadsafe'

# Try to import GPU support
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Try to import ONNX for interoperability 
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Performance monitoring
class PerformanceMonitor:
    """Monitor performance of operations"""
    
    def __init__(self):
        self.operation_times = {}
        self.call_counts = {}
        self._lock = threading.Lock()
    
    def record_operation(self, op_name: str, duration: float):
        """Record operation timing"""
        with self._lock:
            if op_name not in self.operation_times:
                self.operation_times[op_name] = []
                self.call_counts[op_name] = 0
            
            self.operation_times[op_name].append(duration)
            self.call_counts[op_name] += 1
    
    def get_stats(self, op_name: str) -> Dict[str, float]:
        """Get operation statistics"""
        if op_name not in self.operation_times:
            return {}
        
        times = self.operation_times[op_name]
        return {
            'mean_time': np.mean(times),
            'total_time': np.sum(times),
            'call_count': self.call_counts[op_name],
            'min_time': np.min(times),
            'max_time': np.max(times)
        }

def performance_decorator(op_name: str):
    """Decorator to monitor operation performance"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Record performance if monitor exists
            if hasattr(args[0], '_perf_monitor') and args[0]._perf_monitor:
                args[0]._perf_monitor.record_operation(op_name, end_time - start_time)
            
            return result
        return wrapper
    return decorator


class DeviceManager:
    """Device management system inspired by PyTorch"""
    
    def __init__(self):
        self.current_device = 'cpu'
        self.available_devices = ['cpu']
        
        if HAS_GPU:
            try:
                cp.cuda.runtime.getDeviceCount()
                self.available_devices.append('cuda')
            except:
                pass
    
    def set_device(self, device: str) -> None:
        """Set current device"""
        if device not in self.available_devices:
            raise ValueError(f"Device {device} not available. Available: {self.available_devices}")
        self.current_device = device
    
    def get_device(self) -> str:
        """Get current device"""
        return self.current_device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return 'cuda' in self.available_devices


class MemoryPool:
    """Memory pool for efficient allocation"""
    
    def __init__(self):
        self.cpu_pool = {}
        self.gpu_pool = {} if HAS_GPU else None
        
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype, device: str) -> np.ndarray:
        """Allocate memory from pool"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        if device == 'cpu':
            return self._allocate_cpu(shape, dtype, size)
        elif device == 'cuda' and HAS_GPU:
            return self._allocate_gpu(shape, dtype, size)
        else:
            raise ValueError(f"Unsupported device: {device}")
    
    def _allocate_cpu(self, shape: Tuple[int, ...], dtype: np.dtype, size: int) -> np.ndarray:
        """Allocate CPU memory"""
        return np.empty(shape, dtype=dtype)
    
    def _allocate_gpu(self, shape: Tuple[int, ...], dtype: np.dtype, size: int) -> np.ndarray:
        """Allocate GPU memory"""
        if not HAS_GPU:
            raise RuntimeError("GPU not available")
        return cp.empty(shape, dtype=dtype)


# JIT compiled operations for performance
@jit(nopython=True)
def _matmul_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized CPU matrix multiplication"""
    return np.dot(a, b)


@jit(nopython=True, parallel=True)
def _add_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized CPU addition"""
    return a + b


@jit(nopython=True, parallel=True)
def _subtract_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized CPU subtraction"""
    return a - b


@jit(nopython=True, parallel=True)
def _multiply_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized CPU element-wise multiplication"""
    return a * b


@jit(nopython=True, parallel=True)
def _divide_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized CPU element-wise division"""
    return a / b


@jit(nopython=True, parallel=True)
def _conv2d_cpu(input_data: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Optimized CPU 2D convolution"""
    batch_size, in_channels, in_height, in_width = input_data.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1
    
    output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=input_data.dtype)
    
    for b in prange(batch_size):
        for oc in prange(out_channels):
            for oh in prange(out_height):
                for ow in prange(out_width):
                    h_start = oh * stride
                    w_start = ow * stride
                    
                    for ic in range(in_channels):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                output[b, oc, oh, ow] += (
                                    input_data[b, ic, h_start + kh, w_start + kw] * 
                                    kernel[oc, ic, kh, kw]
                                )
    
    return output


@jit(nopython=True, parallel=True)
def _relu_cpu(x: np.ndarray) -> np.ndarray:
    """Optimized CPU ReLU activation"""
    return np.maximum(0, x)


@jit(nopython=True, parallel=True)
def _sigmoid_cpu(x: np.ndarray) -> np.ndarray:
    """Optimized CPU Sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-x))


@jit(nopython=True, parallel=True)
def _tanh_cpu(x: np.ndarray) -> np.ndarray:
    """Optimized CPU Tanh activation"""
    return np.tanh(x)


@jit(nopython=True)
def _power_cpu(x: np.ndarray, power: float) -> np.ndarray:
    """Optimized CPU power operation"""
    return np.power(x, power)


@jit(nopython=True, parallel=True)
def _exp_cpu(x: np.ndarray) -> np.ndarray:
    """Optimized CPU exponential"""
    return np.exp(x)


@jit(nopython=True, parallel=True)
def _log_cpu(x: np.ndarray) -> np.ndarray:
    """Optimized CPU logarithm"""
    return np.log(x)


class NativeBackend:
    """Native backend implementation with optimizations from TF and PyTorch"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.memory_pool = MemoryPool()
        self.jit_cache = {}  # Cache for compiled operations
        self._perf_monitor = PerformanceMonitor()  # Performance monitor
    
    def set_device(self, device: str) -> None:
        """Set current device"""
        self.device_manager.set_device(device)
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device_manager.get_device()
    
    def to_device(self, array: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """Move array to specified device"""
        if device is None:
            device = self.get_device()
        
        if device == 'cpu':
            if HAS_GPU and isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
            return array
        elif device == 'cuda':
            if not HAS_GPU:
                warnings.warn("CUDA not available, using CPU")
                return array
            if isinstance(array, np.ndarray):
                return cp.asarray(array)
            return array
        else:
            raise ValueError(f"Unsupported device: {device}")
    
    @performance_decorator("matmul")
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            result = cp.dot(a_gpu, b_gpu)
            return result
        else:
            return _matmul_cpu(a, b)
    
    @performance_decorator("add")
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu + b_gpu
        else:
            return _add_cpu(a, b)
    
    @performance_decorator("subtract")
    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise subtraction with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu - b_gpu
        else:
            return _subtract_cpu(a, b)
    
    @performance_decorator("multiply")
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu * b_gpu
        else:
            return _multiply_cpu(a, b)
    
    @performance_decorator("divide")
    def divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise division with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu / b_gpu
        else:
            return _divide_cpu(a, b)
    
    @performance_decorator("conv2d")
    def conv2d(self, input_data: np.ndarray, kernel: np.ndarray, 
               stride: int = 1, padding: int = 0) -> np.ndarray:
        """2D convolution with device optimization"""
        device = self.get_device()
        
        # Apply padding if needed
        if padding > 0:
            input_data = np.pad(input_data, 
                              ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                              mode='constant')
        
        if device == 'cuda' and HAS_GPU:
            # Use CuPy's optimized convolution
            input_gpu = self.to_device(input_data, 'cuda')
            kernel_gpu = self.to_device(kernel, 'cuda')
            # Simplified convolution for demo - in practice use optimized CUDA kernels
            return self._conv2d_gpu_simple(input_gpu, kernel_gpu, stride)
        else:
            return _conv2d_cpu(input_data, kernel, stride)
    
    def _conv2d_gpu_simple(self, input_data, kernel, stride):
        """Simplified GPU convolution using CuPy"""
        if not HAS_GPU:
            raise RuntimeError("GPU not available")
        
        # Use CuPy's correlate function for convolution
        from cupyx.scipy import ndimage
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = kernel.shape
        
        out_height = (in_height - kernel_height) // stride + 1
        out_width = (in_width - kernel_width) // stride + 1
        
        output = cp.zeros((batch_size, out_channels, out_height, out_width), dtype=input_data.dtype)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for ic in range(in_channels):
                    # Perform correlation (convolution with flipped kernel)
                    corr = ndimage.correlate(input_data[b, ic], kernel[oc, ic], mode='constant')
                    # Extract valid region with stride
                    output[b, oc] += corr[::stride, ::stride][:out_height, :out_width]
        
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.maximum(0, x_gpu)
        else:
            return _relu_cpu(x)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return 1.0 / (1.0 + cp.exp(-x_gpu))
        else:
            return _sigmoid_cpu(x)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.tanh(x_gpu)
        else:
            return _tanh_cpu(x)
    
    def power(self, x: np.ndarray, power: float) -> np.ndarray:
        """Element-wise power"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.power(x_gpu, power)
        else:
            return _power_cpu(x, power)
    
    def exp(self, x: np.ndarray) -> np.ndarray:
        """Element-wise exponential"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.exp(x_gpu)
        else:
            return _exp_cpu(x)
    
    def log(self, x: np.ndarray) -> np.ndarray:
        """Element-wise logarithm"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.log(x_gpu)
        else:
            return _log_cpu(x)
    
    def zeros(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Create zero tensor"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            return cp.zeros(shape, dtype=dtype)
        else:
            return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Create ones tensor"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            return cp.ones(shape, dtype=dtype)
        else:
            return np.ones(shape, dtype=dtype)
    
    def randn(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Create random normal tensor"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            return cp.random.randn(*shape).astype(dtype)
        else:
            return np.random.randn(*shape).astype(dtype)
    
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu * b_gpu
        else:
            return _multiply_cpu(a, b)
    
    def divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise division with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu / b_gpu
        else:
            return _divide_cpu(a, b)
    
    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise subtraction with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            a_gpu = self.to_device(a, 'cuda')
            b_gpu = self.to_device(b, 'cuda')
            return a_gpu - b_gpu
        else:
            return _subtract_cpu(a, b)
    
    def power(self, x: np.ndarray, power: float) -> np.ndarray:
        """Power operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.power(x_gpu, power)
        else:
            return _power_cpu(x, power)
    
    def exp(self, x: np.ndarray) -> np.ndarray:
        """Exponential operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.exp(x_gpu)
        else:
            return _exp_cpu(x)
    
    def log(self, x: np.ndarray) -> np.ndarray:
        """Logarithm operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.log(x_gpu)
        else:
            return _log_cpu(x)
    
    def sum(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        """Sum operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.sum(x_gpu, axis=axis, keepdims=keepdims)
        else:
            return np.sum(x, axis=axis, keepdims=keepdims)
    
    def mean(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        """Mean operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.mean(x_gpu, axis=axis, keepdims=keepdims)
        else:
            return np.mean(x, axis=axis, keepdims=keepdims)
    
    def max(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        """Max operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.max(x_gpu, axis=axis, keepdims=keepdims)
        else:
            return np.max(x, axis=axis, keepdims=keepdims)
    
    def min(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        """Min operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.min(x_gpu, axis=axis, keepdims=keepdims)
        else:
            return np.min(x, axis=axis, keepdims=keepdims)
    
    def transpose(self, x: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Transpose operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.transpose(x_gpu, axes)
        else:
            return np.transpose(x, axes)
    
    def reshape(self, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.reshape(x_gpu, shape)
        else:
            return np.reshape(x, shape)
    
    def clip(self, x: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
        """Clip operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            return cp.clip(x_gpu, min_val, max_val)
        else:
            return np.clip(x, min_val, max_val)
    
    def gather(self, params: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
        """Gather operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            params_gpu = self.to_device(params, 'cuda')
            indices_gpu = self.to_device(indices, 'cuda')
            return cp.take(params_gpu, indices_gpu, axis=axis)
        else:
            return np.take(params, indices, axis=axis)
    
    def scatter(self, dst: np.ndarray, indices: np.ndarray, updates: np.ndarray, axis: int = 0) -> np.ndarray:
        """Scatter operation with device optimization"""
        device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            dst_gpu = self.to_device(dst, 'cuda')
            indices_gpu = self.to_device(indices, 'cuda')
            updates_gpu = self.to_device(updates, 'cuda')
            cp.scatter(dst_gpu, indices_gpu, updates_gpu, axis=axis)
            return dst_gpu
        else:
            np.scatter(dst, indices, updates, axis=axis)
            return dst
    
    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, 
                   eps: float = 1e-5, device: Optional[str] = None) -> np.ndarray:
        """Layer normalization with device optimization"""
        if device is None:
            device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            weight_gpu = self.to_device(weight, 'cuda')
            bias_gpu = self.to_device(bias, 'cuda')
            
            # CuPy implementation
            from cupyx.nn import LayerNorm
            ln = LayerNorm(x_gpu.shape[1:], eps=eps)
            return ln(x_gpu, weight_gpu, bias_gpu)
        else:
            # Numpy implementation
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + eps)
            return x_norm * weight + bias
    
    def batch_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, 
                   running_mean: np.ndarray, running_var: np.ndarray, 
                   momentum: float = 0.9, eps: float = 1e-5, 
                   training: bool = True, device: Optional[str] = None) -> np.ndarray:
        """Batch normalization with device optimization"""
        if device is None:
            device = self.get_device()
        
        if device == 'cuda' and HAS_GPU:
            x_gpu = self.to_device(x, 'cuda')
            weight_gpu = self.to_device(weight, 'cuda')
            bias_gpu = self.to_device(bias, 'cuda')
            running_mean_gpu = self.to_device(running_mean, 'cuda')
            running_var_gpu = self.to_device(running_var, 'cuda')
            
            if training:
                # CuPy implementation
                from cupyx.nn import BatchNorm2d
                bn = BatchNorm2d(x_gpu.shape[1], eps=eps, momentum=momentum, training=True)
                return bn(x_gpu, weight_gpu, bias_gpu)
            else:
                # Use running mean and variance
                return (x_gpu - running_mean_gpu) / cp.sqrt(running_var_gpu + eps)
        else:
            # Numpy implementation
            if training:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
                # Update running mean and variance
                running_mean = momentum * running_mean + (1 - momentum) * mean
                running_var = momentum * running_var + (1 - momentum) * var
            else:
                mean = running_mean
                var = running_var
            
            x_norm = (x - mean) / np.sqrt(var + eps)
            return x_norm * weight + bias


# Global backend instance
backend = NativeBackend()