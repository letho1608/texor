from typing import Optional, Union, Any, Dict
import numpy as np
import tensorflow as tf
import torch
import torch.jit
from ..core.backend import backend

class AutoCompiler:
    """Automatically compile models/functions for optimal performance"""
    
    def __init__(self, model: Any, input_spec: Optional[Dict[str, Any]] = None):
        self.model = model
        self.input_spec = input_spec
        self.compiled_tf = None
        self.compiled_torch = None
        self.best_backend = None
        
    def compile(self) -> None:
        """Compile model for both backends"""
        # Try TensorFlow compilation
        try:
            self._compile_tensorflow()
        except Exception as e:
            print(f"TensorFlow compilation failed: {e}")
            
        # Try PyTorch compilation
        try:
            self._compile_pytorch()
        except Exception as e:
            print(f"PyTorch compilation failed: {e}")
            
        # Benchmark and select best backend
        self._benchmark()
        
    def _compile_tensorflow(self) -> None:
        """Compile using TensorFlow"""
        if hasattr(self.model, 'tensorflow'):
            tf_model = self.model.tensorflow()
            
            # Convert to TF function
            @tf.function(jit_compile=True)
            def compiled_func(*args, **kwargs):
                return tf_model(*args, **kwargs)
                
            self.compiled_tf = compiled_func
            
    def _compile_pytorch(self) -> None:
        """Compile using PyTorch TorchScript"""
        if hasattr(self.model, 'pytorch'):
            torch_model = self.model.pytorch()
            
            # Create example inputs based on input_spec
            if self.input_spec:
                example_inputs = self._create_example_inputs()
                self.compiled_torch = torch.jit.trace(torch_model, example_inputs)
            else:
                self.compiled_torch = torch.jit.script(torch_model)
                
    def _benchmark(self, num_runs: int = 100) -> None:
        """Benchmark both compiled versions"""
        if not (self.compiled_tf or self.compiled_torch):
            raise RuntimeError("No successful compilation")
            
        example_inputs = self._create_example_inputs()
        tf_times = []
        torch_times = []
        
        # Benchmark TensorFlow
        if self.compiled_tf:
            # Warmup
            for _ in range(10):
                self.compiled_tf(example_inputs)
                
            for _ in range(num_runs):
                start = tf.timestamp()
                self.compiled_tf(example_inputs)
                tf_times.append(float(tf.timestamp() - start))
                
        # Benchmark PyTorch
        if self.compiled_torch:
            torch_inputs = torch.from_numpy(example_inputs.numpy())
            
            # Warmup
            for _ in range(10):
                self.compiled_torch(torch_inputs)
                
            for _ in range(num_runs):
                start = time.time()
                self.compiled_torch(torch_inputs)
                torch_times.append(time.time() - start)
                
        # Select best backend
        if tf_times and torch_times:
            tf_mean = np.mean(tf_times)
            torch_mean = np.mean(torch_times)
            self.best_backend = 'tensorflow' if tf_mean < torch_mean else 'pytorch'
        elif tf_times:
            self.best_backend = 'tensorflow'
        else:
            self.best_backend = 'pytorch'
            
    def _create_example_inputs(self) -> Union[tf.Tensor, torch.Tensor]:
        """Create example inputs based on input_spec"""
        if not self.input_spec:
            raise ValueError("Input specification required for compilation")
            
        # Create numpy arrays based on spec
        example_inputs = {
            name: np.random.randn(*spec['shape']).astype(spec.get('dtype', np.float32))
            for name, spec in self.input_spec.items()
        }
        
        # Convert to tensor based on current backend
        if backend.current == 'tensorflow':
            return {k: tf.convert_to_tensor(v) for k, v in example_inputs.items()}
        else:
            return {k: torch.from_numpy(v) for k, v in example_inputs.items()}
            
    def __call__(self, *args, **kwargs):
        """Run the compiled model"""
        if self.best_backend == 'tensorflow' and self.compiled_tf:
            return self.compiled_tf(*args, **kwargs)
        elif self.best_backend == 'pytorch' and self.compiled_torch:
            return self.compiled_torch(*args, **kwargs)
        else:
            raise RuntimeError("No compiled model available")

class AutoOptimizer:
    """Automatically optimize model execution"""
    
    def __init__(self, model: Any):
        self.model = model
        self.compiled_model = None
        self.mixed_precision = False
        self.xla_enabled = False
        
    def optimize(self, enable_mixed_precision: bool = True, 
                enable_xla: bool = True) -> None:
        """Apply various optimization techniques"""
        self.mixed_precision = enable_mixed_precision
        self.xla_enabled = enable_xla
        
        # Enable mixed precision if requested
        if enable_mixed_precision:
            backend.enable_mixed_precision()
            
        # Enable XLA for TensorFlow
        if enable_xla and backend.current == 'tensorflow':
            tf.config.optimizer.set_jit(True)
            
        # Compile model
        input_spec = self._infer_input_spec()
        compiler = AutoCompiler(self.model, input_spec)
        compiler.compile()
        self.compiled_model = compiler
        
    def _infer_input_spec(self) -> Dict[str, Any]:
        """Infer input specification from model"""
        # Try to get input shape from model
        if hasattr(self.model, 'input_shape'):
            return {'input': {'shape': self.model.input_shape}}
            
        # Default to some common input shapes
        return {'input': {'shape': (1, 3, 224, 224), 'dtype': np.float32}}
        
    def __call__(self, *args, **kwargs):
        """Run the optimized model"""
        if self.compiled_model is None:
            raise RuntimeError("Model not optimized. Call optimize() first")
        return self.compiled_model(*args, **kwargs)

def auto_compile(model: Any, input_spec: Optional[Dict[str, Any]] = None) -> AutoCompiler:
    """Utility function to quickly compile a model"""
    compiler = AutoCompiler(model, input_spec)
    compiler.compile()
    return compiler

def auto_optimize(model: Any) -> AutoOptimizer:
    """Utility function to quickly optimize a model"""
    optimizer = AutoOptimizer(model)
    optimizer.optimize()
    return optimizer