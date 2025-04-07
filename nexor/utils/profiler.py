from typing import Optional, Callable, Dict, Any
import time
import cProfile
import pstats
import io
import numpy as np
import tensorflow as tf
import torch
from ..core.backend import backend

class OperationProfiler:
    """Profile operations across different backends"""
    
    def __init__(self):
        self.results = {}
        
    def profile_operation(self, operation: str, 
                         input_data: Any,
                         tf_func: Callable,
                         torch_func: Callable,
                         repeat: int = 100) -> Dict[str, float]:
        """Profile same operation on both backends"""
        results = {}
        
        # Profile TensorFlow
        tf_tensor = tf.convert_to_tensor(input_data)
        tf_times = []
        
        # Warmup
        for _ in range(5):
            tf_func(tf_tensor)
            
        for _ in range(repeat):
            start = time.perf_counter()
            tf_func(tf_tensor)
            tf_times.append(time.perf_counter() - start)
            
        results['tensorflow'] = {
            'mean': np.mean(tf_times),
            'std': np.std(tf_times),
            'min': np.min(tf_times),
            'max': np.max(tf_times)
        }
        
        # Profile PyTorch
        torch_tensor = torch.from_numpy(input_data)
        torch_times = []
        
        # Warmup
        for _ in range(5):
            torch_func(torch_tensor)
            
        for _ in range(repeat):
            start = time.perf_counter()
            torch_func(torch_tensor)
            torch_times.append(time.perf_counter() - start)
            
        results['pytorch'] = {
            'mean': np.mean(torch_times),
            'std': np.std(torch_times),
            'min': np.min(torch_times),
            'max': np.max(torch_times)
        }
        
        self.results[operation] = results
        return results
        
    def suggest_backend(self, operation: str) -> str:
        """Suggest best backend based on profiling results"""
        if operation not in self.results:
            return 'auto'
            
        tf_time = self.results[operation]['tensorflow']['mean']
        torch_time = self.results[operation]['pytorch']['mean']
        
        return 'tensorflow' if tf_time < torch_time else 'pytorch'
        
    def print_summary(self) -> None:
        """Print profiling summary"""
        print("\nOperation Profiling Summary")
        print("-" * 50)
        
        for op, results in self.results.items():
            print(f"\nOperation: {op}")
            print("TensorFlow:")
            tf_results = results['tensorflow']
            print(f"  Mean: {tf_results['mean']*1000:.3f} ms")
            print(f"  Std:  {tf_results['std']*1000:.3f} ms")
            print(f"  Min:  {tf_results['min']*1000:.3f} ms")
            print(f"  Max:  {tf_results['max']*1000:.3f} ms")
            
            print("PyTorch:")
            torch_results = results['pytorch']
            print(f"  Mean: {torch_results['mean']*1000:.3f} ms")
            print(f"  Std:  {torch_results['std']*1000:.3f} ms")
            print(f"  Min:  {torch_results['min']*1000:.3f} ms")
            print(f"  Max:  {torch_results['max']*1000:.3f} ms")
            
            suggested = self.suggest_backend(op)
            print(f"Suggested backend: {suggested}")

class MemoryProfiler:
    """Profile memory usage across different backends"""
    
    def __init__(self):
        self.baseline = None
        self.current = None
        
    def start(self) -> None:
        """Start memory profiling"""
        self.baseline = backend.memory_status()
        
    def end(self) -> Dict[str, float]:
        """End memory profiling and return usage"""
        self.current = backend.memory_status()
        
        usage = {'tensorflow': {}, 'pytorch': {}}
        
        # Calculate TensorFlow memory usage
        for device in self.current['tensorflow']:
            if device in self.baseline['tensorflow']:
                baseline = self.baseline['tensorflow'][device].get('allocated', 0)
                current = self.current['tensorflow'][device].get('allocated', 0)
                usage['tensorflow'][device] = current - baseline
                
        # Calculate PyTorch memory usage
        for device in self.current['pytorch']:
            if device in self.baseline['pytorch']:
                baseline = self.baseline['pytorch'][device].get('allocated', 0)
                current = self.current['pytorch'][device].get('allocated', 0)
                usage['pytorch'][device] = current - baseline
                
        return usage
        
    def print_usage(self, usage: Dict[str, Dict[str, float]]) -> None:
        """Print memory usage summary"""
        print("\nMemory Usage Summary")
        print("-" * 50)
        
        print("\nTensorFlow:")
        for device, mem in usage['tensorflow'].items():
            print(f"  {device}: {mem:.2f} MB")
            
        print("\nPyTorch:")
        for device, mem in usage['pytorch'].items():
            print(f"  {device}: {mem:.2f} MB")

class CodeProfiler:
    """Profile Python code execution"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        
    def start(self) -> None:
        """Start code profiling"""
        self.profiler.enable()
        
    def end(self) -> None:
        """End code profiling"""
        self.profiler.disable()
        
    def print_stats(self, sort_by: str = 'cumtime', lines: int = 20) -> None:
        """Print profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(lines)
        print(s.getvalue())
        
def profile_operation(operation: str,
                     input_data: Any,
                     tf_func: Callable,
                     torch_func: Callable,
                     repeat: int = 100) -> Dict[str, float]:
    """Utility function to quickly profile an operation"""
    profiler = OperationProfiler()
    return profiler.profile_operation(operation, input_data, tf_func, torch_func, repeat)

def memory_profile():
    """Context manager for memory profiling"""
    profiler = MemoryProfiler()
    profiler.start()
    return profiler

def code_profile():
    """Context manager for code profiling"""
    profiler = CodeProfiler()
    profiler.start()
    return profiler