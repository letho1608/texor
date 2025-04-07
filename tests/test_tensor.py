import unittest
import numpy as np
import tensorflow as tf
import torch
from nexor.core.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_creation(self):
        """Test tensor creation from different data sources"""
        # From numpy array
        np_data = np.array([[1, 2], [3, 4]])
        t1 = Tensor(np_data)
        self.assertTrue(np.array_equal(t1.numpy(), np_data))
        
        # From TensorFlow tensor
        tf_data = tf.constant([[1, 2], [3, 4]])
        t2 = Tensor(tf_data)
        self.assertTrue(np.array_equal(t2.numpy(), tf_data.numpy()))
        
        # From PyTorch tensor
        torch_data = torch.tensor([[1, 2], [3, 4]])
        t3 = Tensor(torch_data)
        self.assertTrue(np.array_equal(t3.numpy(), torch_data.detach().numpy()))
        
        # From Python list
        list_data = [[1, 2], [3, 4]]
        t4 = Tensor(list_data)
        self.assertTrue(np.array_equal(t4.numpy(), np.array(list_data)))

    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        
        # Addition
        c = a + b
        self.assertTrue(np.array_equal(
            c.numpy(),
            np.array([[6, 8], [10, 12]])
        ))
        
        # Multiplication
        d = a * b
        self.assertTrue(np.array_equal(
            d.numpy(),
            np.array([[5, 12], [21, 32]])
        ))
        
        # Matrix multiplication
        e = a @ b
        self.assertTrue(np.array_equal(
            e.numpy(),
            np.array([[19, 22], [43, 50]])
        ))

    def test_gradients(self):
        """Test gradient computation"""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x * x  # Element-wise square
        z = y.sum()  # Sum all elements
        
        z.backward()
        # Gradient should be 2x
        expected_grad = np.array([[2, 4], [6, 8]], dtype=np.float32)
        actual_grad = x.grad
        print(f"\nExpected grad:\n{expected_grad}")
        print(f"Actual grad:\n{actual_grad}")
        print(f"Shapes: expected {expected_grad.shape}, actual {actual_grad.shape}")
        print(f"Types: expected {expected_grad.dtype}, actual {actual_grad.dtype}")
        
        if actual_grad is not None:
            diff = np.abs(actual_grad - expected_grad)
            print(f"Max absolute difference: {np.max(diff)}")
            
        self.assertTrue(np.allclose(actual_grad, expected_grad))

    def test_backend_conversion(self):
        """Test conversion between different backends"""
        x = Tensor([[1, 2], [3, 4]])
        
        # Test TensorFlow conversion
        tf_tensor = x.tensorflow()
        self.assertIsInstance(tf_tensor, tf.Tensor)
        self.assertTrue(np.array_equal(tf_tensor.numpy(), x.numpy()))
        
        # Test PyTorch conversion
        torch_tensor = x.pytorch()
        self.assertIsInstance(torch_tensor, torch.Tensor)
        self.assertTrue(np.array_equal(torch_tensor.detach().numpy(), x.numpy()))

    def test_shape_and_dtype(self):
        """Test shape and dtype properties"""
        x = Tensor(np.random.randn(2, 3, 4))
        
        self.assertEqual(x.shape, (2, 3, 4))
        self.assertEqual(x.dtype, np.float64)

if __name__ == '__main__':
    unittest.main()