from typing import Optional, Union, Any
import numpy as np
import tensorflow as tf
import torch

class Backend:
    """Backend manager for hybrid computation"""
    
    def __init__(self):
        self._current = 'auto'  # 'auto', 'tensorflow', 'pytorch'
        self._tf_eager = tf.executing_eagerly()
        self._torch_grad = torch.is_grad_enabled()
        
    @property
    def current(self) -> str:
        return self._current
        
    def set_backend(self, name: str) -> None:
        if name not in ['auto', 'tensorflow', 'pytorch']:
            raise ValueError(f"Unsupported backend: {name}")
        self._current = name
        
    def optimize_backend(self, operation: str, 
                        input_size: Optional[tuple] = None,
                        dtype: Optional[str] = None) -> str:
        """Automatically choose the best backend for given operation"""
        if self._current != 'auto':
            return self._current
            
        # Optimization rules based on operation type and data
        if operation in ['conv2d', 'max_pool2d', 'batch_norm']:
            # TF is generally faster for CNNs
            return 'tensorflow'
            
        elif operation in ['lstm', 'gru', 'rnn']:
            # PyTorch handles RNNs better
            return 'pytorch'
            
        elif operation == 'transformer':
            # PyTorch's transformer implementation is more flexible
            return 'pytorch'
            
        elif operation == 'linear':
            if input_size and input_size[0] > 1000:
                # TF handles large batch matrix multiply better
                return 'tensorflow'
            return 'pytorch'
            
        return 'pytorch'  # Default to PyTorch for other cases
        
    def convert_tensor(self, data: Any, target_backend: str) -> Union[tf.Tensor, torch.Tensor]:
        """Convert tensor between backends"""
        # Handle Nexor Tensor
        if hasattr(data, '_numpy'):  # Check if it's Nexor Tensor
            data = data._numpy

        if isinstance(data, np.ndarray):
            if target_backend == 'tensorflow':
                return tf.convert_to_tensor(data)
            else:
                return torch.from_numpy(data)
                
        elif isinstance(data, tf.Tensor):
            if target_backend == 'tensorflow':
                return data
            return torch.from_numpy(data.numpy())
            
        elif isinstance(data, torch.Tensor):
            if target_backend == 'pytorch':
                return data
            return tf.convert_to_tensor(data.detach().cpu().numpy())
            
        raise ValueError(f"Unsupported data type: {type(data)}")

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication using optimal backend"""
        from ..core.tensor import Tensor
        from ..core.autograd import MatMulBackward
        backend = self.optimize_backend('linear', a.shape)
        
        requires_grad = getattr(a, 'requires_grad', False) or getattr(b, 'requires_grad', False)
        
        # Convert inputs to numpy if needed
        a_np = a._numpy if hasattr(a, '_numpy') else a
        b_np = b._numpy if hasattr(b, '_numpy') else b
        
        # Compute result using numpy
        result = np.matmul(a_np, b_np)
        
        # Create output tensor
        output = Tensor(result, requires_grad=requires_grad)
        
        # Add context for backprop if needed
        if requires_grad:
            if isinstance(a, Tensor):
                a.requires_grad = True
            if isinstance(b, Tensor):
                b.requires_grad = True
            output._ctx = MatMulBackward(a, b)
            
        return output
            
        output = Tensor(result, requires_grad=requires_grad)
        if requires_grad:
            output.ctx = MatMulBackward(a, b)
        return output

    def max_pool2d(self, inputs: Any, kernel_size: tuple,
                  stride: tuple, padding: tuple) -> Any:
        """2D Max pooling using optimal backend"""
        from ..core.tensor import Tensor
        backend = self.optimize_backend('max_pool2d', inputs.shape)
        
        if backend == 'tensorflow':
            x = self.convert_tensor(inputs, 'tensorflow')
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
            result = tf.nn.max_pool2d(
                x,
                ksize=[1, kernel_size[0], kernel_size[1], 1],
                strides=[1, stride[0], stride[1], 1],
                padding='SAME' if padding[0] > 0 else 'VALID'
            ).numpy()
            result = np.transpose(result, [0, 3, 1, 2])  # NHWC -> NCHW
        else:
            x = self.convert_tensor(inputs, 'pytorch')
            result = torch.nn.functional.max_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).detach().numpy()
            
        return Tensor(result)

    def conv2d(self, inputs: Any, weight: Any, bias: Any,
              stride: tuple, padding: tuple) -> Any:
        """2D Convolution using optimal backend"""
        from ..core.tensor import Tensor
        backend = self.optimize_backend('conv2d', inputs.shape)
        
        if backend == 'tensorflow':
            x = self.convert_tensor(inputs, 'tensorflow')
            w = self.convert_tensor(weight, 'tensorflow')
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
            w = tf.transpose(w, [2, 3, 1, 0])  # OIHW -> HWIO
            
            result = tf.nn.conv2d(
                x,
                w,
                strides=[1, stride[0], stride[1], 1],
                padding='SAME' if padding[0] > 0 else 'VALID'
            ).numpy()
            
            if bias is not None:
                b = self.convert_tensor(bias, 'tensorflow')
                b_np = b.numpy()
                result = result + b_np.reshape(1, 1, 1, -1)
                
            result = np.transpose(result, [0, 3, 1, 2])  # NHWC -> NCHW
        else:
            x = self.convert_tensor(inputs, 'pytorch')
            w = self.convert_tensor(weight, 'pytorch')
            b = self.convert_tensor(bias, 'pytorch') if bias is not None else None
            result = torch.nn.functional.conv2d(x, w, b, stride=stride, padding=padding)
            result = result.detach().numpy()
            
        return Tensor(result)

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition using optimal backend"""
        from ..core.tensor import Tensor
        from ..core.autograd import AddBackward
        
        requires_grad = getattr(a, 'requires_grad', False) or getattr(b, 'requires_grad', False)
        
        # Convert to numpy arrays
        a_np = a._numpy if hasattr(a, '_numpy') else a
        b_np = b._numpy if hasattr(b, '_numpy') else b
        
        # Compute result
        result = a_np + b_np
        
        # Create output tensor
        output = Tensor(result, requires_grad=requires_grad)
        
        # Add context for backprop if needed
        if requires_grad:
            if isinstance(a, Tensor):
                a.requires_grad = True
            if isinstance(b, Tensor):
                b.requires_grad = True
            output._ctx = AddBackward(a, b)
            
        return output

    def div(self, x: Any, scalar: float) -> Any:
        """Element-wise division of tensor by scalar"""
        from ..core.tensor import Tensor
        
        if hasattr(x, '_numpy'):
            x = x._numpy
            
        if isinstance(x, np.ndarray):
            return Tensor(x / scalar)
            
        backend = self.current
        if backend == 'tensorflow':
            result = tf.divide(x, scalar).numpy()
        else:
            result = x.div(scalar).detach().numpy()
            
        return Tensor(result)
        
    def get_device(self, backend: Optional[str] = None) -> str:
        """Get optimal device for given backend"""
        if backend is None:
            backend = self.current
            
        if backend == 'tensorflow':
            return 'GPU:0' if tf.test.is_gpu_available() else 'CPU:0'
        else:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
            
    def memory_status(self) -> dict:
        """Get memory usage status for both backends"""
        status = {}
        
        # TensorFlow memory status
        tf_status = {}
        for device in tf.config.list_physical_devices():
            try:
                mem_info = tf.config.experimental.get_memory_info(device.name)
                tf_status[device.name] = {
                    'allocated': mem_info['current'] / (1024**2),  # MB
                    'peak': mem_info['peak'] / (1024**2)  # MB
                }
            except:
                tf_status[device.name] = {'error': 'Memory info not available'}
        status['tensorflow'] = tf_status
        
        # PyTorch memory status
        torch_status = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch_status[f'cuda:{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / (1024**2),  # MB
                    'cached': torch.cuda.memory_reserved(i) / (1024**2)  # MB
                }
        status['pytorch'] = torch_status
        
        return status
        
    def clear_memory(self) -> None:
        """Clear memory for both backends"""
        # Clear TensorFlow memory
        if tf.test.is_gpu_available():
            tf.keras.backend.clear_session()
            
        # Clear PyTorch memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for both backends"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training for both backends"""
        # TensorFlow mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # PyTorch mixed precision
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    def get_backend_info(self) -> dict:
        """Get information about available backends"""
        info = {
            'tensorflow': {
                'version': tf.__version__,
                'eager_execution': tf.executing_eagerly(),
                'devices': [device.name for device in tf.config.list_physical_devices()],
                'gpu_available': tf.test.is_gpu_available()
            },
            'pytorch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        return info

# Global backend manager
backend = Backend()

def set_backend(name: str) -> None:
    backend.set_backend(name)

def get_current_backend() -> str:
    return backend.current

def optimize_for_operation(operation: str, input_size: Optional[tuple] = None,
                         dtype: Optional[str] = None) -> str:
    return backend.optimize_backend(operation, input_size, dtype)

def get_optimal_device() -> str:
    return backend.get_device()

def memory_status() -> dict:
    return backend.memory_status()

def clear_memory() -> None:
    backend.clear_memory()

def set_random_seed(seed: int) -> None:
    backend.set_random_seed(seed)

def enable_mixed_precision() -> None:
    backend.enable_mixed_precision()

def get_backend_info() -> dict:
    return backend.get_backend_info()