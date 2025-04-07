import unittest
import numpy as np
import tensorflow as tf
import torch
from nexor.core.tensor import Tensor
from nexor.core.backend import backend

class TestTensorAdvanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Save original backend state
        cls.original_backend = backend.current
        backend.set_backend('auto')

    @classmethod
    def tearDownClass(cls):
        """Restore original backend state"""
        backend.set_backend(cls.original_backend)

    def test_complex_gradients(self):
        """Test gradient computation with complex operations"""
        x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        y = Tensor([[2., 1.], [1., 2.]], requires_grad=True)
        
        # Complex computation: z = sin(x) * y + exp(x * y)
        z = (x.sin() * y + (x * y).exp()).sum()
        z.backward()
        
        # Calculate expected gradients using numpy
        x_np = x.numpy()
        y_np = y.numpy()
        expected_x_grad = np.cos(x_np) * y_np + y_np * np.exp(x_np * y_np)
        expected_y_grad = np.sin(x_np) + x_np * np.exp(x_np * y_np)
        
        self.assertTrue(np.allclose(x.grad, expected_x_grad, rtol=1e-5))
        self.assertTrue(np.allclose(y.grad, expected_y_grad, rtol=1e-5))

    def test_error_handling(self):
        """Test error handling in backend conversion"""
        # Test invalid data type
        with self.assertRaises(ValueError):
            Tensor("invalid data")
            
        # Test shape mismatch in operations
        a = Tensor([[1, 2]])
        b = Tensor([[1], [2]])
        
        with self.assertRaises(ValueError):
            c = a + b
            
        # Test invalid backend name
        with self.assertRaises(ValueError):
            backend.set_backend('invalid_backend')

    def test_mixed_precision(self):
        """Test mixed precision operations"""
        # Enable mixed precision
        backend.enable_mixed_precision()
        
        x = Tensor([[1., 2.], [3., 4.]])
        y = Tensor([[5., 6.], [7., 8.]])
        
        # Test if operations maintain precision
        z = x @ y
        self.assertEqual(z.dtype, np.float32)  # Should be float32 in mixed precision
        
        # Test if large numbers don't overflow
        large_x = Tensor([[1e4, 2e4], [3e4, 4e4]])
        large_y = Tensor([[5e4, 6e4], [7e4, 8e4]])
        large_z = large_x @ large_y
        
        expected = np.array([[1.9e9, 2.2e9], [4.3e9, 5.0e9]])
        self.assertTrue(np.allclose(large_z.numpy(), expected, rtol=1e-3))

    def test_device_placement(self):
        """Test tensor operations on different devices"""
        x = Tensor([[1., 2.], [3., 4.]])
        
        # Test CPU operations
        backend.set_backend('pytorch')
        cpu_tensor = x.pytorch()
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Test GPU operations if available
        if torch.cuda.is_available():
            try:
                gpu_tensor = cpu_tensor.cuda()
                self.assertEqual(gpu_tensor.device.type, 'cuda')
                # Test operations on GPU
                result = gpu_tensor @ gpu_tensor
                self.assertEqual(result.device.type, 'cuda')
            except Exception as e:
                self.fail(f"GPU operations failed: {e}")

    def test_memory_management(self):
        """Test memory management and cleanup"""
        import gc
        import sys
        
        def create_large_tensor():
            return Tensor(np.random.randn(1000, 1000))
            
        initial_count = len(gc.get_objects())
        x = create_large_tensor()
        mid_count = len(gc.get_objects())
        
        del x
        gc.collect()
        final_count = len(gc.get_objects())
        
        # Check if tensor was properly cleaned up
        self.assertLess(final_count - initial_count, mid_count - initial_count)

    def test_backend_optimization(self):
        """Test automatic backend optimization"""
        # Test convolution operation
        x = Tensor(np.random.randn(1, 3, 32, 32))
        w = Tensor(np.random.randn(16, 3, 3, 3))
        
        backend.set_backend('auto')
        optimal_backend = backend.optimize_backend('conv2d', input_size=x.shape)
        
        # TensorFlow should be chosen for convolutions
        self.assertEqual(optimal_backend, 'tensorflow')
        
        # Test matrix multiplication
        x = Tensor(np.random.randn(1000, 1000))
        y = Tensor(np.random.randn(1000, 1000))
        
        optimal_backend = backend.optimize_backend('linear', input_size=x.shape)
        
        # TensorFlow should be chosen for large matrix multiplications
        self.assertEqual(optimal_backend, 'tensorflow')

    def test_tensor_conversion_consistency(self):
        """Test consistency of tensor conversions between backends"""
        original = Tensor(np.random.randn(10, 10))
        
        # Convert to TensorFlow
        tf_tensor = original.tensorflow()
        back_from_tf = Tensor(tf_tensor)
        
        # Convert to PyTorch
        torch_tensor = original.pytorch()
        back_from_torch = Tensor(torch_tensor)
        
        # Check if conversions maintain values
        self.assertTrue(np.allclose(original.numpy(), back_from_tf.numpy()))
        self.assertTrue(np.allclose(original.numpy(), back_from_torch.numpy()))
        
        # Check if gradients are preserved
        x = Tensor(np.random.randn(5, 5), requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        
        grad_numpy = x.grad.copy()
        
        # Convert to TensorFlow and back
        x_tf = x.tensorflow()
        x_back = Tensor(x_tf)
        self.assertTrue(np.allclose(x_back.grad, grad_numpy))
        
        # Convert to PyTorch and back
        x_torch = x.pytorch()
        x_back = Tensor(x_torch)
        self.assertTrue(np.allclose(x_back.grad, grad_numpy))

if __name__ == '__main__':
    unittest.main()