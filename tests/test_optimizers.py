import unittest
import numpy as np
from nexor.core import Tensor
from nexor.optim import SGD, Adam, RMSprop, Adagrad, Adadelta

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        self.params = [
            Tensor(np.random.randn(10, 5), requires_grad=True),
            Tensor(np.random.randn(5), requires_grad=True)
        ]
        
        # Create a simple quadratic loss: f(x) = x^2
        # Gradient should be: f'(x) = 2x
        self.initial_values = [p.numpy().copy() for p in self.params]
        
    def compute_gradient(self):
        """Compute gradients for test parameters"""
        for param, init_val in zip(self.params, self.initial_values):
            param.grad = Tensor(2 * init_val)  # Gradient of x^2 is 2x

    def test_sgd_optimizer(self):
        """Test SGD optimizer with and without momentum"""
        # Test vanilla SGD
        optimizer = SGD(self.params, lr=0.1)
        
        # First update
        self.compute_gradient()
        optimizer.step()
        
        # Check if parameters were updated correctly
        for param, init_val in zip(self.params, self.initial_values):
            expected = init_val - 0.1 * 2 * init_val
            self.assertTrue(np.allclose(param.numpy(), expected))
            
        # Test SGD with momentum
        self.params = [
            Tensor(np.random.randn(10, 5), requires_grad=True),
            Tensor(np.random.randn(5), requires_grad=True)
        ]
        self.initial_values = [p.numpy().copy() for p in self.params]
        
        optimizer = SGD(self.params, lr=0.1, momentum=0.9)
        
        # First update
        self.compute_gradient()
        optimizer.step()
        
        # Velocities should be initialized and updated
        for v in optimizer.velocities:
            self.assertFalse(np.all(v == 0))

    def test_adam_optimizer(self):
        """Test Adam optimizer"""
        optimizer = Adam(self.params, lr=0.001, betas=(0.9, 0.999))
        
        # Multiple updates to test momentum and variance
        for _ in range(3):
            self.compute_gradient()
            optimizer.step()
            
        # Check if momentum and variance terms are being updated
        for m in optimizer.m:
            self.assertFalse(np.all(m == 0))
        for v in optimizer.v:
            self.assertFalse(np.all(v == 0))
            
        # Parameters should be updated
        for param, init_val in zip(self.params, self.initial_values):
            self.assertFalse(np.array_equal(param.numpy(), init_val))

    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer"""
        optimizer = RMSprop(self.params, lr=0.01, alpha=0.99)
        
        # Multiple updates
        for _ in range(3):
            self.compute_gradient()
            optimizer.step()
            
        # Check if running averages are being updated
        for avg in optimizer.square_avg:
            self.assertFalse(np.all(avg == 0))

    def test_adagrad_optimizer(self):
        """Test Adagrad optimizer"""
        optimizer = Adagrad(self.params, lr=0.01)
        
        # Multiple updates
        for _ in range(3):
            self.compute_gradient()
            optimizer.step()
            
        # Check if accumulated gradients are increasing
        for state in optimizer.state:
            self.assertTrue(np.all(state >= 0))
            self.assertFalse(np.all(state == 0))

    def test_adadelta_optimizer(self):
        """Test Adadelta optimizer"""
        optimizer = Adadelta(self.params, rho=0.9)
        
        # Multiple updates
        for _ in range(3):
            self.compute_gradient()
            optimizer.step()
            
        # Check if running averages are being updated
        for avg in optimizer.square_avg:
            self.assertFalse(np.all(avg == 0))
        for delta in optimizer.acc_delta:
            self.assertFalse(np.all(delta == 0))

    def test_zero_grad(self):
        """Test gradient zeroing functionality"""
        optimizer = SGD(self.params, lr=0.1)
        
        # Set some gradients
        self.compute_gradient()
        
        # Zero out gradients
        optimizer.zero_grad()
        
        # Check if all gradients are zero
        for param in self.params:
            self.assertTrue(np.all(param.grad.numpy() == 0))

if __name__ == '__main__':
    unittest.main()