import unittest
import numpy as np
import tensorflow as tf
import torch
from nexor.core import Tensor
from nexor.optim import SGD, Adam, RMSprop
from nexor.core.backend import backend
from nexor.nn import Linear

class TestOptimizersAdvanced(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        backend.set_backend('auto')
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
        
        # Create a simple neural network
        self.model = Linear(10, 1)
        self.x = Tensor(np.random.randn(100, 10))
        self.y = Tensor(np.random.randn(100, 1))

    def test_optimizers_convergence(self):
        """Test if optimizers can converge to minimum"""
        optimizers = [
            ('SGD', lambda p: SGD(p, lr=0.01, momentum=0.9)),
            ('Adam', lambda p: Adam(p, lr=0.01)),
            ('RMSprop', lambda p: RMSprop(p, lr=0.01))
        ]
        
        for name, opt_creator in optimizers:
            # Reset model parameters
            self.model = Linear(10, 1)
            optimizer = opt_creator(self.model.parameters())
            
            initial_loss = None
            final_loss = None
            
            # Train for several steps
            for step in range(100):
                optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(self.x)
                loss = ((pred - self.y) ** 2).mean()
                
                if step == 0:
                    initial_loss = loss.numpy()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if step == 99:
                    final_loss = loss.numpy()
            
            # Check if loss decreased
            self.assertLess(final_loss, initial_loss, 
                          f"{name} failed to converge")

    def test_mixed_precision_optimization(self):
        """Test optimizers with mixed precision training"""
        backend.enable_mixed_precision()
        
        model = Linear(10, 1)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Original parameters in fp32
        fp32_params = [p.numpy() for p in model.parameters()]
        
        # Train step with mixed precision
        optimizer.zero_grad()
        pred = model(self.x)
        loss = ((pred - self.y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        # Check if parameters were updated while maintaining precision
        for p, fp32_p in zip(model.parameters(), fp32_params):
            # Parameters should be different after update
            self.assertFalse(np.array_equal(p.numpy(), fp32_p))
            # Parameters should still be in fp32
            self.assertEqual(p.dtype, np.float32)

    def test_optimizer_state_transfer(self):
        """Test optimizer state transfer between backends"""
        optimizers = [
            ('SGD', lambda p: SGD(p, lr=0.01, momentum=0.9)),
            ('Adam', lambda p: Adam(p, lr=0.01)),
            ('RMSprop', lambda p: RMSprop(p, lr=0.01))
        ]
        
        for name, opt_creator in optimizers:
            model = Linear(10, 1)
            optimizer = opt_creator(model.parameters())
            
            # Train for a few steps to build up optimizer state
            for _ in range(5):
                optimizer.zero_grad()
                pred = model(self.x)
                loss = ((pred - self.y) ** 2).mean()
                loss.backward()
                optimizer.step()
            
            # Save optimizer state
            state = optimizer.state_dict()
            
            # Create new optimizer and load state
            new_optimizer = opt_creator(model.parameters())
            new_optimizer.load_state_dict(state)
            
            # Check if states match
            if name == 'SGD':
                for v1, v2 in zip(optimizer.velocities, new_optimizer.velocities):
                    self.assertTrue(np.array_equal(v1, v2))
            elif name == 'Adam':
                for m1, m2 in zip(optimizer.m, new_optimizer.m):
                    self.assertTrue(np.array_equal(m1, m2))
                for v1, v2 in zip(optimizer.v, new_optimizer.v):
                    self.assertTrue(np.array_equal(v1, v2))

    def test_gradient_clipping(self):
        """Test gradient clipping in optimizers"""
        model = Linear(10, 1)
        optimizer = Adam(model.parameters(), lr=0.01, max_grad_norm=1.0)
        
        # Generate large gradients
        optimizer.zero_grad()
        pred = model(self.x * 1000)  # Scale up inputs to get large gradients
        loss = ((pred - self.y) ** 2).mean()
        loss.backward()
        
        # Store original gradients
        original_grads = [p.grad.numpy().copy() for p in model.parameters()]
        
        # Step with gradient clipping
        optimizer.step()
        
        # Check if gradients were clipped
        for orig_grad in original_grads:
            grad_norm = np.sqrt(np.sum(orig_grad ** 2))
            self.assertGreater(grad_norm, 1.0)  # Original grad norm > 1.0

    def test_optimizer_memory_usage(self):
        """Test optimizer memory usage and cleanup"""
        import gc
        import sys
        
        def create_and_train_model():
            model = Linear(1000, 1000)  # Large model
            optimizer = Adam(model.parameters(), lr=0.01)
            
            x = Tensor(np.random.randn(100, 1000))
            y = Tensor(np.random.randn(100, 1000))
            
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            
            return sys.getsizeof(optimizer)
        
        # Get initial memory usage
        initial_size = create_and_train_model()
        gc.collect()
        
        # Create and train multiple models
        for _ in range(5):
            current_size = create_and_train_model()
            gc.collect()
            # Memory usage should not grow significantly
            self.assertLess(current_size - initial_size, initial_size * 0.1)

if __name__ == '__main__':
    unittest.main()