#!/usr/bin/env python3
"""
Auto debugger for Texor library.
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
from texor.core.native_tensor import Tensor
from texor.nn.layers import Linear
from texor.nn.activations import ReLU
from texor.nn.model import Sequential

class AutoDebugger:
    def __init__(self):
        self.test_results = {}
        self.profiling_results = {}
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
            'tests.test_model'
        ]
        
        for module_name in test_modules:
            self._run_test_module(module_name)
            
        print("\n=== Running Integration Tests ===")
        self._run_integration_tests()
        
    def _run_test_module(self, module_name):
        """Run a specific test module"""
        try:
            print(f"\nRunning {module_name}...")
            
            # Import the module
            test_module = __import__(module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(suite)
            
            # Store results
            self.test_results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful(),
                'output': stream.getvalue()
            }
            
            if result.wasSuccessful():
                print(f"✓ {module_name}: {result.testsRun} tests passed")
            else:
                print(f"✗ {module_name}: {result.failures} failures, {result.errors} errors")
                
        except Exception as e:
            print(f"✗ Error running {module_name}: {str(e)}")
            self.test_results[module_name] = {
                'error': str(e),
                'success': False
            }
            
    def _run_integration_tests(self):
        """Run integration tests"""
        try:
            from tests.run_integration_tests import run_integration_tests
            success = run_integration_tests()
            self.test_results['integration'] = {'success': success}
            
            if success:
                print("✓ Integration tests passed")
            else:
                print("✗ Integration tests failed")
                
        except Exception as e:
            print(f"✗ Error running integration tests: {str(e)}")
            self.test_results['integration'] = {'error': str(e), 'success': False}
            
    def profile_performance(self):
        """Profile native operations"""
        print("\n=== Profiling Performance ===")
        
        operations = {
            'tensor_creation': self._profile_tensor_creation,
            'matrix_multiply': self._profile_matrix_multiply,
            'convolution': self._profile_convolution,
            'gradient_computation': self._profile_gradients
        }
        
        for name, func in operations.items():
            print(f"\nProfiling {name}...")
            try:
                profiler = cProfile.Profile()
                profiler.enable()
                
                execution_time = func()
                
                profiler.disable()
                
                # Store profiling results
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(10)  # Top 10 functions
                
                self.profiling_results[name] = {
                    'execution_time': execution_time,
                    'profile_output': s.getvalue()
                }
                
                print(f"  Execution time: {execution_time:.4f}s")
                
            except Exception as e:
                print(f"  Error profiling {name}: {str(e)}")
                self.profiling_results[name] = {'error': str(e)}
                
    def _profile_tensor_creation(self):
        """Profile tensor creation and basic operations"""
        start_time = time.time()
        
        # Create various tensors
        for i in range(100):
            x = Tensor(np.random.randn(100, 100))
            y = x + x
            z = x * y
            result = z.sum()
            
        return time.time() - start_time
        
    def _profile_matrix_multiply(self):
        """Profile matrix multiplication"""
        start_time = time.time()
        
        x = Tensor(np.random.randn(500, 500))
        y = Tensor(np.random.randn(500, 500))
        
        for _ in range(10):
            result = x @ y
            
        return time.time() - start_time
        
    def _profile_convolution(self):
        """Profile convolution operations"""
        start_time = time.time()
        
        from texor.nn.layers import Conv2D
        conv = Conv2D(3, 16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(10, 3, 32, 32))
        
        for _ in range(20):
            result = conv(x)
            
        return time.time() - start_time
        
    def _profile_gradients(self):
        """Profile gradient computation"""
        start_time = time.time()
        
        model = Sequential([
            Linear(100, 50),
            ReLU(),
            Linear(50, 10)
        ])
        
        x = Tensor(np.random.randn(32, 100))
        
        for _ in range(50):
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            # Clear gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
                    
        return time.time() - start_time
        
    def check_memory_usage(self):
        """Check for potential memory issues"""
        print("\n=== Checking Memory Usage ===")
        
        try:
            # Test for memory leaks in repeated operations
            print("Testing for memory leaks...")
            
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run memory-intensive operations
            for i in range(50):
                x = Tensor(np.random.randn(200, 200))
                y = Tensor(np.random.randn(200, 200))
                z = x @ y
                loss = z.sum()
                loss.backward()
                
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"  Iteration {i}, Memory: {current_memory:.1f} MB")
                    
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")
            
            if memory_increase > 100:  # More than 100MB increase
                self.issues_found.append(f"Potential memory leak: {memory_increase:.1f} MB increase")
                print("⚠️  Potential memory leak detected")
            else:
                print("✓ No significant memory leaks detected")
                
        except ImportError:
            print("psutil not available, skipping detailed memory analysis")
        except Exception as e:
            print(f"Error checking memory usage: {str(e)}")
            
    def check_numerical_stability(self):
        """Check for numerical stability issues"""
        print("\n=== Checking Numerical Stability ===")
        
        issues = []
        
        # Test for NaN/Inf in operations
        print("Testing for NaN/Inf values...")
        
        try:
            # Large values
            x = Tensor(np.ones((10, 10)) * 1e6)
            y = x * x
            if np.any(np.isnan(y.data)) or np.any(np.isinf(y.data)):
                issues.append("NaN/Inf detected in large value operations")
                
            # Small values
            x = Tensor(np.ones((10, 10)) * 1e-10)
            y = x * x
            if np.any(np.isnan(y.data)) or np.any(np.isinf(y.data)):
                issues.append("NaN/Inf detected in small value operations")
                
            # Division by very small numbers
            x = Tensor(np.ones((10, 10)))
            y = Tensor(np.ones((10, 10)) * 1e-15)
            z = x / y
            if np.any(np.isnan(z.data)) or np.any(np.isinf(z.data)):
                issues.append("NaN/Inf detected in division operations")
                
            if not issues:
                print("✓ No numerical stability issues detected")
            else:
                for issue in issues:
                    print(f"⚠️  {issue}")
                    self.issues_found.append(issue)
                    
        except Exception as e:
            print(f"Error checking numerical stability: {str(e)}")
            
    def generate_report(self):
        """Generate comprehensive debug report"""
        print("\n" + "="*60)
        print("TEXOR AUTO DEBUG REPORT")
        print("="*60)
        
        # Test Results Summary
        print("\n--- Test Results Summary ---")
        total_modules = len(self.test_results)
        successful_modules = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        print(f"Test modules run: {total_modules}")
        print(f"Successful modules: {successful_modules}")
        print(f"Failed modules: {total_modules - successful_modules}")
        
        for module, result in self.test_results.items():
            if result.get('success', False):
                print(f"  ✓ {module}")
            else:
                print(f"  ✗ {module}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                    
        # Performance Summary
        print("\n--- Performance Summary ---")
        for operation, result in self.profiling_results.items():
            if 'execution_time' in result:
                print(f"  {operation}: {result['execution_time']:.4f}s")
            else:
                print(f"  {operation}: Error")
                
        # Issues Found
        print("\n--- Issues Found ---")
        if self.issues_found:
            for issue in self.issues_found:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✓ No issues detected")
            
        # Recommendations
        print("\n--- Recommendations ---")
        if successful_modules == total_modules and not self.issues_found:
            print("  ✓ All tests passing, no issues detected")
            print("  ✓ Texor is functioning correctly")
        else:
            print("  → Review failed tests and fix issues")
            print("  → Consider memory optimization if leaks detected")
            print("  → Check numerical stability in edge cases")
            
        print("\n" + "="*60)

def main():
    """Main debug function"""
    print("Starting Texor Auto Debug...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create debugger
    debugger = AutoDebugger()
    
    # Run all checks
    debugger.run_all_tests()
    debugger.profile_performance()
    debugger.check_memory_usage()
    debugger.check_numerical_stability()
    
    # Generate report
    debugger.generate_report()
    
    print("\nAuto debug complete!")

if __name__ == '__main__':
    main()
