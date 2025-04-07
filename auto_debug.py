#!/usr/bin/env python3
"""
Auto debugger for Nexor library.
Automatically runs tests, profiles performance, and identifies potential issues.
"""

import os
import sys
import time
import unittest
import cProfile
import pstats
import io
import traceback
import numpy as np
import tensorflow as tf
import torch
from nexor.core.backend import backend
from nexor.utils.profiler import OperationProfiler, MemoryProfiler
from nexor.core import Tensor
from nexor.nn import Sequential, Linear, ReLU

class AutoDebugger:
    def __init__(self):
        self.test_results = {}
        self.profiling_results = {}
        self.memory_usage = {}
        self.issues_found = []
        
    def run_all_tests(self):
        """Run all test suites and collect results"""
        print("\n=== Running All Tests ===")
        
        test_modules = [
            'tests.test_tensor',
            'tests.test_tensor_advanced',
            'tests.test_layers',
            'tests.test_layers_advanced',
            'tests.test_optimizers',
            'tests.test_optimizers_advanced',
            'tests.test_model',
            'tests.test_model_advanced'
        ]
        
        for module_name in test_modules:
            self._run_test_module(module_name)
            
        print("\n=== Running Integration Tests ===")
        self._run_integration_tests()
        
    def _run_test_module(self, module_name):
        """Run a specific test module"""
        try:
            module = __import__(module_name, fromlist=[''])
            suite = unittest.TestLoader().loadTestsFromModule(module)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            self.test_results[module_name] = {
                'success': result.wasSuccessful(),
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped)
            }
            
            if not result.wasSuccessful():
                self._analyze_test_failures(result)
                
        except Exception as e:
            self.issues_found.append(f"Error running {module_name}: {str(e)}")
            
    def _run_integration_tests(self):
        """Run integration tests"""
        try:
            from tests.run_integration_tests import IntegrationTests
            suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTests)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            self.test_results['integration'] = {
                'success': result.wasSuccessful(),
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped)
            }
            
        except Exception as e:
            self.issues_found.append(f"Error running integration tests: {str(e)}")
            
    def _analyze_test_failures(self, result):
        """Analyze test failures and suggest fixes"""
        for failure in result.failures + result.errors:
            test_case = failure[0]
            error_message = failure[1]
            
            # Extract relevant information
            test_name = test_case.id().split('.')[-1]
            error_type = error_message.split('\n')[0]
            
            # Analyze common issues
            if "AssertionError" in error_type:
                self._analyze_assertion_error(test_name, error_message)
            elif "RuntimeError" in error_type:
                self._analyze_runtime_error(test_name, error_message)
            elif "ValueError" in error_type:
                self._analyze_value_error(test_name, error_message)
                
    def profile_performance(self):
        """Profile library performance"""
        print("\n=== Profiling Performance ===")
        
        profiler = OperationProfiler()
        
        # Profile basic operations
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        
        operations = {
            'matrix_multiply': (
                lambda t: tf.matmul(t, t),
                lambda t: torch.matmul(t, t)
            ),
            'convolution': (
                lambda t: tf.nn.conv2d(t[None, ..., None], tf.random.normal([3, 3, 1, 1]), 1, 'SAME'),
                lambda t: torch.nn.functional.conv2d(t[None, None], torch.randn(1, 1, 3, 3), padding=1)
            )
        }
        
        for op_name, (tf_func, torch_func) in operations.items():
            results = profiler.profile_operation(
                op_name,
                x,
                tf_func,
                torch_func
            )
            self.profiling_results[op_name] = results
            
    def monitor_memory(self):
        """Monitor memory usage"""
        print("\n=== Monitoring Memory Usage ===")
        
        memory_profiler = MemoryProfiler()
        memory_profiler.start()
        
        # Create and train a model to monitor memory
        model = Sequential([
            Linear(1000, 100),
            ReLU(),
            Linear(100, 10)
        ])
        
        x = Tensor(np.random.randn(1000, 1000))
        y = Tensor(np.random.randn(1000, 10))
        
        # Train for a few steps
        for _ in range(10):
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            
        self.memory_usage = memory_profiler.end()
        
    def generate_report(self):
        """Generate debug report"""
        print("\n=== Debug Report ===")
        
        # Test results summary
        print("\nTest Results:")
        for module, results in self.test_results.items():
            print(f"\n{module}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
                
        # Performance summary
        print("\nPerformance Results:")
        for op, results in self.profiling_results.items():
            print(f"\n{op}:")
            for backend, metrics in results.items():
                print(f"  {backend}:")
                print(f"    Mean time: {metrics['mean']*1000:.3f} ms")
                print(f"    Std dev:   {metrics['std']*1000:.3f} ms")
                
        # Memory usage summary
        print("\nMemory Usage:")
        for backend, devices in self.memory_usage.items():
            print(f"\n{backend}:")
            for device, usage in devices.items():
                print(f"  {device}: {usage:.2f} MB")
                
        # Issues found
        if self.issues_found:
            print("\nIssues Found:")
            for issue in self.issues_found:
                print(f"- {issue}")
                
    def suggest_optimizations(self):
        """Suggest possible optimizations"""
        print("\n=== Optimization Suggestions ===")
        
        # Analyze backend performance
        for op, results in self.profiling_results.items():
            tf_time = results['tensorflow']['mean']
            torch_time = results['pytorch']['mean']
            
            if tf_time < torch_time:
                print(f"\n{op}: Consider using TensorFlow backend (faster by {((torch_time/tf_time)-1)*100:.1f}%)")
            else:
                print(f"\n{op}: Consider using PyTorch backend (faster by {((tf_time/torch_time)-1)*100:.1f}%)")
                
        # Check memory usage
        if self.memory_usage:
            for backend, devices in self.memory_usage.items():
                for device, usage in devices.items():
                    if usage > 1000:  # More than 1GB
                        print(f"\nHigh memory usage detected on {backend} {device}: {usage:.2f} MB")
                        print("Consider implementing memory optimization techniques:")
                        print("- Gradient checkpointing")
                        print("- Model parallelism")
                        print("- Gradient accumulation")

def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)
    
    debugger = AutoDebugger()
    
    try:
        # Run tests and collect data
        debugger.run_all_tests()
        debugger.profile_performance()
        debugger.monitor_memory()
        
        # Generate reports
        debugger.generate_report()
        debugger.suggest_optimizations()
        
    except Exception as e:
        print(f"\nError during auto-debugging: {str(e)}")
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
