import unittest
import numpy as np
from texor.core.native_tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_creation(self):
        """Test tensor creation from different data sources"""
        # From numpy array
        np_data = np.array([[1, 2], [3, 4]])
        t1 = Tensor(np_data)
        self.assertTrue(np.array_equal(t1.data, np_data))
        
        # From Python list
        list_data = [[1, 2], [3, 4]]
        t4 = Tensor(list_data)
        self.assertTrue(np.array_equal(t4.data, np.array(list_data)))

    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        
        # Addition
        c = a + b
        self.assertTrue(np.array_equal(
            c.data,
            np.array([[6, 8], [10, 12]])
        ))
        
        # Multiplication
        d = a * b
        self.assertTrue(np.array_equal(
            d.data,
            np.array([[5, 12], [21, 32]])
        ))
        
        # Matrix multiplication
        e = a @ b
        expected = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(e.data, expected))

    def test_gradients(self):
        """Test gradient computation"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x * x  # Element-wise square
        z = y.sum()  # Sum all elements
        
        z.backward()
        # Gradient should be 2x
        expected_grad = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
        actual_grad = x.grad.data if x.grad is not None else None
        
        self.assertIsNotNone(actual_grad)
        self.assertTrue(np.allclose(actual_grad, expected_grad))

    def test_shape_and_dtype(self):
        """Test shape and dtype properties"""
        x = Tensor(np.random.randn(2, 3, 4))
        
        self.assertEqual(x.shape, (2, 3, 4))
        self.assertEqual(len(x.shape), 3)

    def test_device_management(self):
        """Test device handling"""
        x = Tensor([[1, 2], [3, 4]])
        self.assertIn(x.device, ['cpu', 'cuda:0'])

    def test_requires_grad(self):
        """Test gradient requirement setting"""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        self.assertTrue(x.requires_grad)
        
        y = Tensor([[1, 2], [3, 4]], requires_grad=False)
        self.assertFalse(y.requires_grad)

    def test_tensor_operations(self):
        """Test various tensor operations"""
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        
        # Test sum
        s = x.sum()
        self.assertEqual(s.data.item(), 21)
        
        # Test mean
        m = x.mean()
        self.assertEqual(m.data.item(), 3.5)

if __name__ == '__main__':
    unittest.main()