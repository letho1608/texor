import unittest
import numpy as np
from nexor.core import Tensor
from nexor.nn import Sequential, Linear, ReLU, MSELoss
from nexor.optim import SGD

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up a simple model for testing"""
        self.model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])
        
        # Create sample data
        self.x = Tensor(np.random.randn(32, 10))  # 32 samples, 10 features
        self.y = Tensor(np.random.randn(32, 1))   # 32 samples, 1 target
        
    def test_model_creation(self):
        """Test model creation and architecture"""
        # Check layer count
        self.assertEqual(len(self.model.layers), 3)
        
        # Check layer types
        self.assertIsInstance(self.model.layers[0], Linear)
        self.assertIsInstance(self.model.layers[1], ReLU)
        self.assertIsInstance(self.model.layers[2], Linear)
        
        # Check layer dimensions
        self.assertEqual(self.model.layers[0].in_features, 10)
        self.assertEqual(self.model.layers[0].out_features, 5)
        self.assertEqual(self.model.layers[2].in_features, 5)
        self.assertEqual(self.model.layers[2].out_features, 1)

    def test_forward_pass(self):
        """Test forward pass through the model"""
        # Single sample
        single_input = Tensor(np.random.randn(1, 10))
        output = self.model(single_input)
        self.assertEqual(output.shape, (1, 1))
        
        # Batch of samples
        batch_input = Tensor(np.random.randn(16, 10))
        output = self.model(batch_input)
        self.assertEqual(output.shape, (16, 1))

    def test_model_compilation(self):
        """Test model compilation with different optimizers and losses"""
        self.model.compile(
            optimizer='sgd',
            loss='mse'
        )
        
        # Check if optimizer and loss function are set correctly
        self.assertIsInstance(self.model.optimizer, SGD)
        self.assertIsInstance(self.model.loss_fn, MSELoss)

    def test_model_training(self):
        """Test model training functionality"""
        self.model.compile(
            optimizer='sgd',
            loss='mse'
        )
        
        # Train for a few epochs
        history = self.model.fit(
            x=self.x,
            y=self.y,
            epochs=3,
            batch_size=8
        )
        
        # Check if history contains loss values
        self.assertTrue('loss' in history)
        self.assertEqual(len(history['loss']), 3)
        
        # Check if loss is decreasing
        self.assertTrue(history['loss'][0] > history['loss'][-1])

    def test_model_evaluation(self):
        """Test model evaluation modes"""
        # Training mode
        self.model.train()
        self.assertTrue(self.model.training)
        for layer in self.model.layers:
            self.assertTrue(getattr(layer, 'training', True))
            
        # Evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
        for layer in self.model.layers:
            self.assertFalse(getattr(layer, 'training', False))

    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Single prediction
        single_input = Tensor(np.random.randn(1, 10))
        pred = self.model.predict(single_input)
        self.assertEqual(pred.shape, (1, 1))
        
        # Batch prediction
        batch_input = Tensor(np.random.randn(16, 10))
        preds = self.model.predict(batch_input)
        self.assertEqual(preds.shape, (16, 1))

    def test_parameter_access(self):
        """Test access to model parameters"""
        params = self.model.parameters()
        
        # Check if we have parameters for both Linear layers
        # Each Linear layer has weights and biases
        self.assertEqual(len(params), 4)
        
        # Check parameter shapes
        self.assertEqual(params[0].shape, (10, 5))  # First layer weights
        self.assertEqual(params[1].shape, (5,))     # First layer bias
        self.assertEqual(params[2].shape, (5, 1))   # Second layer weights
        self.assertEqual(params[3].shape, (1,))     # Second layer bias

    def test_validation_split(self):
        """Test training with validation split"""
        self.model.compile(
            optimizer='sgd',
            loss='mse'
        )
        
        history = self.model.fit(
            x=self.x,
            y=self.y,
            epochs=3,
            batch_size=8,
            validation_split=0.2
        )
        
        # Check if validation metrics are present
        self.assertTrue('val_loss' in history)
        self.assertEqual(len(history['val_loss']), 3)

if __name__ == '__main__':
    unittest.main()