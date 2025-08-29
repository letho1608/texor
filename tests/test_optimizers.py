import unittest
import numpy as np
from texor.core.native_tensor import Tensor
from texor.optim.optimizers import SGD, Adam, RMSprop

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        self.params = [
            Tensor(np.random.randn(10, 5), requires_grad=True),
            Tensor(np.random.randn(5), requires_grad=True)
        ]
        
        # Create simple gradients for testing
        for param in self.params:
            param.grad = Tensor(np.ones_like(param.data) * 0.1)
        
    def test_sgd_optimizer(self):
        """Test SGD optimizer with and without momentum"""
        # Test vanilla SGD
        optimizer = SGD(self.params, lr=0.1)
        
        initial_values = [p.data.copy() for p in self.params]
        
        # First update
        optimizer.step()
        
        # Check if parameters were updated correctly
        for param, init_val in zip(self.params, initial_values):
            expected = init_val - 0.1 * 0.1  # lr * grad
            self.assertTrue(np.allclose(param.data, expected))
            
        # Test SGD with momentum
        self.setUp()  # Reset parameters
        optimizer = SGD(self.params, lr=0.1, momentum=0.9)
        
        # First update
        optimizer.step()
        
        # Velocities should be initialized
        self.assertEqual(len(optimizer.velocities), len(self.params))

    def test_adam_optimizer(self):
        """Test Adam optimizer"""
        optimizer = Adam(self.params, lr=0.001)
        
        initial_values = [p.data.copy() for p in self.params]
        
        # Perform several steps
        for _ in range(3):
            optimizer.step()
            
        # Parameters should have been updated
        for param, init_val in zip(self.params, initial_values):
            self.assertFalse(np.array_equal(param.data, init_val))
            
        # Check internal state
        self.assertEqual(optimizer.t, 3)  # Should have done 3 steps
        self.assertEqual(len(optimizer.m), len(self.params))
        self.assertEqual(len(optimizer.v), len(self.params))

    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer"""
        optimizer = RMSprop(self.params, lr=0.01, alpha=0.99)
        
        initial_values = [p.data.copy() for p in self.params]
        
        # Perform several steps
        for _ in range(3):
            optimizer.step()
            
        # Parameters should have been updated
        for param, init_val in zip(self.params, initial_values):
            self.assertFalse(np.array_equal(param.data, init_val))
            
        # Check internal state
        self.assertEqual(len(optimizer.square_avg), len(self.params))

    def test_zero_grad(self):
        """Test gradient zeroing functionality"""
        # Set some gradients
        for param in self.params:
            param.grad = Tensor(np.ones_like(param.data))
            
        optimizer = SGD(self.params, lr=0.1)
        optimizer.zero_grad()
        
        # All gradients should be zero
        for param in self.params:
            if param.grad is not None:
                self.assertTrue(np.allclose(param.grad.data, 0))

    def test_state_dict(self):
        """Test optimizer state dictionary save/load"""
        optimizer = Adam(self.params, lr=0.001)
        
        # Perform some steps to create state
        for _ in range(2):
            optimizer.step()
            
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = Adam(self.params, lr=0.01)  # Different lr
        new_optimizer.load_state_dict(state)
          # Check that state was loaded correctly
        self.assertEqual(new_optimizer.lr, 0.001)  # Should be from saved state
        self.assertEqual(new_optimizer.t, 2)

    def compute_gradient(self):
        """Helper method to compute gradients for testing"""
        for param in self.params:
            param.grad = Tensor(np.random.randn(*param.shape) * 0.1)


if __name__ == '__main__':
    unittest.main()