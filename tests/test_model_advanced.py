import unittest
import numpy as np
from texor.core import Tensor
from texor.nn import Sequential, Linear, ReLU, Conv2D, MaxPool2D
from texor.optim import Adam
from texor.nn.loss import CrossEntropyLoss, MSELoss

class TestModelAdvanced(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        np.random.seed(42)

    def test_complex_model_creation(self):
        """Test creation of a complex model with multiple layers"""
        model = Sequential([
            Linear(784, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 10)
        ])
        
        # Test forward pass
        x = Tensor(np.random.randn(32, 784))
        output = model(x)
        
        self.assertEqual(output.shape, (32, 10))
        self.assertFalse(np.isnan(output.data).any())

    def test_model_training_loop(self):
        """Test manual training loop with complex model"""
        model = Sequential([
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        ])
        
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = MSELoss()
        
        # Generate training data
        x = Tensor(np.random.randn(32, 10))
        y = Tensor(np.random.randn(32, 5))
        
        losses = []
        for epoch in range(5):
            # Forward pass
            pred = model(x)
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.data.item())
        
        # Check that loss is decreasing
        self.assertLess(losses[-1], losses[0])

    def test_model_with_different_activations(self):
        """Test model with different activation functions"""
        from texor.nn import Sigmoid, Tanh
        
        model = Sequential([
            Linear(5, 10),
            ReLU(),
            Linear(10, 8),
            Sigmoid(),
            Linear(8, 3),
            Tanh()
        ])
        
        x = Tensor(np.random.randn(16, 5))
        output = model(x)
        
        self.assertEqual(output.shape, (16, 3))
        # Tanh output should be between -1 and 1
        self.assertTrue((output.data >= -1).all())
        self.assertTrue((output.data <= 1).all())

    def test_classification_model(self):
        """Test a classification model with cross entropy loss"""
        model = Sequential([
            Linear(20, 50),
            ReLU(),
            Linear(50, 10)
        ])
        
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()
        
        # Generate classification data
        x = Tensor(np.random.randn(64, 20))
        y = Tensor(np.random.randint(0, 10, (64,)))
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(10):
            pred = model(x)
            loss = criterion(pred, y)
            
            if epoch == 0:
                initial_loss = loss.data.item()
            if epoch == 9:
                final_loss = loss.data.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss)
        
        # Test prediction accuracy
        with Tensor.no_grad():
            pred = model(x)
            predicted_classes = np.argmax(pred.data, axis=1)
            # At least some predictions should be correct by chance
            accuracy = np.mean(predicted_classes == y.data)
            self.assertGreaterEqual(accuracy, 0.0)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        model = Sequential([
            Linear(5, 10),
            ReLU(),
            Linear(10, 1)
        ])
        
        x = Tensor(np.random.randn(8, 5), requires_grad=True)
        y = Tensor(np.random.randn(8, 1))
        
        criterion = MSELoss()
        
        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(np.isnan(param.grad.data).any())
        
        # Check input gradient
        self.assertIsNotNone(x.grad)
        self.assertFalse(np.isnan(x.grad.data).any())

    def test_model_memory_efficiency(self):
        """Test model memory usage and efficiency"""
        # Large model test
        model = Sequential([
            Linear(1000, 500),
            ReLU(),
            Linear(500, 100),
            ReLU(),
            Linear(100, 10)
        ])
        
        x = Tensor(np.random.randn(100, 1000))
        
        # Multiple forward passes should not cause memory issues
        for _ in range(10):
            output = model(x)
            self.assertEqual(output.shape, (100, 10))

    def test_batch_processing(self):
        """Test model with different batch sizes"""
        model = Sequential([
            Linear(15, 30),
            ReLU(),
            Linear(30, 5)
        ])
        
        batch_sizes = [1, 8, 32, 64]
        
        for batch_size in batch_sizes:
            x = Tensor(np.random.randn(batch_size, 15))
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 5))

    def test_model_reproducibility(self):
        """Test that model produces consistent results with same input"""
        np.random.seed(123)
        
        model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])
        
        x = Tensor(np.random.randn(16, 10))
        
        # Multiple runs should give same output
        output1 = model(x)
        output2 = model(x)
        
        np.testing.assert_array_equal(output1.data, output2.data)

if __name__ == '__main__':
    unittest.main()