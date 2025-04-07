#!/usr/bin/env python3
"""
Integration test runner for Nexor library.
Tests interactions between different components and backends.
"""

import unittest
import sys
import os
import time
import numpy as np
import tensorflow as tf
import torch
from nexor.core import Tensor
from nexor.core.backend import backend
from nexor.nn import Sequential, Linear, ReLU, Conv2D
from nexor.nn.advanced_layers import ResidualBlock, LSTM
from nexor.optim import Adam

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
        
        # Create test data
        cls.x_small = np.random.randn(32, 10)
        cls.y_small = np.random.randn(32, 1)
        cls.x_image = np.random.randn(16, 3, 32, 32)
        cls.y_image = np.eye(10)[np.random.randint(0, 10, 16)]
        
    def setUp(self):
        """Reset backend before each test"""
        backend.set_backend('auto')
        
    def test_end_to_end_training(self):
        """Test complete training pipeline with backend switching"""
        # Create model
        model = Sequential([
            Conv2D(3, 32, kernel_size=3, padding=1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            Linear(64 * 16 * 16, 10)
        ])
        
        # Test with both backends
        for backend_name in ['tensorflow', 'pytorch']:
            backend.set_backend(backend_name)
            
            # Compile model
            model.compile(
                optimizer=Adam(model.parameters(), lr=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Enable mixed precision
            backend.enable_mixed_precision()
            
            # Train
            history = model.fit(
                x=Tensor(self.x_image),
                y=Tensor(self.y_image),
                epochs=3,
                batch_size=4,
                validation_split=0.25
            )
            
            # Verify training results
            self.assertIn('loss', history)
            self.assertIn('accuracy', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_accuracy', history)
            
            # Check memory usage
            memory_status = backend.memory_status()
            print(f"\nMemory status for {backend_name}:")
            print(memory_status)
            
            # Clear memory
            backend.clear_memory()
            
    def test_backend_switching_stability(self):
        """Test stability when switching backends during operations"""
        model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])
        
        x = Tensor(self.x_small)
        y = Tensor(self.y_small)
        
        # Initial prediction with TensorFlow
        backend.set_backend('tensorflow')
        pred1 = model(x)
        
        # Switch to PyTorch
        backend.set_backend('pytorch')
        pred2 = model(x)
        
        # Results should be similar
        self.assertTrue(np.allclose(pred1.numpy(), pred2.numpy(), rtol=1e-5))
        
        # Train with backend switching
        for epoch in range(3):
            # Switch backend each epoch
            backend.set_backend('tensorflow' if epoch % 2 == 0 else 'pytorch')
            
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            
            # Parameters should maintain consistency
            for param in model.parameters():
                self.assertFalse(np.any(np.isnan(param.numpy())))
                
    def test_memory_management(self):
        """Test memory management under heavy load"""
        try:
            for _ in range(5):
                # Create and train large model
                model = Sequential([
                    Linear(1000, 500),
                    ReLU(),
                    Linear(500, 100),
                    ReLU(),
                    Linear(100, 10)
                ])
                
                x = Tensor(np.random.randn(128, 1000))
                y = Tensor(np.eye(10)[np.random.randint(0, 10, 128)])
                
                # Train with both backends
                for backend_name in ['tensorflow', 'pytorch']:
                    backend.set_backend(backend_name)
                    model.compile(optimizer='adam', loss='categorical_crossentropy')
                    model.fit(x, y, epochs=2, batch_size=32)
                    
                    # Force memory cleanup
                    backend.clear_memory()
                    
                # Check memory status
                memory_info = backend.memory_status()
                print("\nMemory status after iteration:")
                print(memory_info)
                
        except Exception as e:
            self.fail(f"Memory management test failed: {str(e)}")
            
    def test_error_handling(self):
        """Test error handling and recovery"""
        model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])
        
        # Test invalid backend
        with self.assertRaises(ValueError):
            backend.set_backend('invalid_backend')
            
        # Test invalid operation
        try:
            x = Tensor(self.x_small)
            y = Tensor(np.random.randn(32, 2))  # Incompatible shape
            loss = ((model(x) - y) ** 2).mean()
        except ValueError as e:
            print(f"Caught expected error: {e}")
        else:
            self.fail("Expected ValueError was not raised")
            
        # Test recovery after error
        x = Tensor(self.x_small)
        y = Tensor(self.y_small)
        loss = ((model(x) - y) ** 2).mean()  # Should work fine
        
    def test_gradient_accumulation(self):
        """Test gradient accumulation across backends"""
        model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])
        
        for backend_name in ['tensorflow', 'pytorch']:
            backend.set_backend(backend_name)
            
            # Reset gradients
            for param in model.parameters():
                param.grad = None
                
            # Accumulate gradients manually
            accumulated_grad = None
            for i in range(4):
                start_idx = i * 8
                end_idx = start_idx + 8
                
                x_batch = Tensor(self.x_small[start_idx:end_idx])
                y_batch = Tensor(self.y_small[start_idx:end_idx])
                
                loss = ((model(x_batch) - y_batch) ** 2).mean()
                loss.backward()
                
                if accumulated_grad is None:
                    accumulated_grad = [p.grad.numpy().copy() for p in model.parameters()]
                else:
                    for acc_grad, param in zip(accumulated_grad, model.parameters()):
                        acc_grad += param.grad.numpy()
                        
            # Check accumulated gradients
            for acc_grad, param in zip(accumulated_grad, model.parameters()):
                self.assertTrue(np.allclose(acc_grad / 4, param.grad.numpy()))

if __name__ == '__main__':
    unittest.main(verbosity=2)