import unittest
import numpy as np
from nexor.core import Tensor
from nexor.nn.layers import Linear, Conv2d, MaxPool2d, Dropout

class TestLayers(unittest.TestCase):
    def test_linear_layer(self):
        """Test Linear layer functionality"""
        batch_size = 32
        in_features = 20
        out_features = 10
        
        # Create layer
        layer = Linear(in_features, out_features)
        
        # Create input
        x = Tensor(np.random.randn(batch_size, in_features))
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))
        
        # Check if weights and bias have correct shapes
        self.assertEqual(layer.weight.shape, (in_features, out_features))
        self.assertEqual(layer.bias.shape, (out_features,))
        
        # Test gradient computation
        if output.requires_grad:
            loss = output.sum()
            loss.backward()
            self.assertIsNotNone(layer.weight.grad)
            self.assertIsNotNone(layer.bias.grad)

    def test_conv2d_layer(self):
        """Test Conv2D layer functionality"""
        batch_size = 16
        in_channels = 3
        out_channels = 64
        height = 32
        width = 32
        kernel_size = 3
        
        # Create layer
        layer = Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        # Create input
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        # Forward pass
        output = layer(x)
        
        # Check output shape (with padding=1, output size should be same as input)
        self.assertEqual(output.shape, 
                        (batch_size, out_channels, height, width))
        
        # Check if weights and bias have correct shapes
        self.assertEqual(layer.weight.shape, 
                        (out_channels, in_channels, kernel_size, kernel_size))
        self.assertEqual(layer.bias.shape, (out_channels,))

    def test_maxpool2d_layer(self):
        """Test MaxPool2D layer functionality"""
        batch_size = 16
        channels = 64
        height = 32
        width = 32
        pool_size = 2
        
        # Create layer
        layer = MaxPool2d(kernel_size=pool_size)
        
        # Create input
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        # Forward pass
        output = layer(x)
        
        # Check output shape (should be halved in spatial dimensions)
        self.assertEqual(output.shape, 
                        (batch_size, channels, height//pool_size, width//pool_size))
        
        # Check if max pooling works correctly
        x_np = x.numpy()
        output_np = output.numpy()
        
        # Check if output values are maximum values from input pooling regions
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height//pool_size):
                    for w in range(width//pool_size):
                        pool_region = x_np[b, c, 
                                         h*pool_size:(h+1)*pool_size, 
                                         w*pool_size:(w+1)*pool_size]
                        self.assertEqual(output_np[b, c, h, w], np.max(pool_region))

    def test_dropout_layer(self):
        """Test Dropout layer functionality"""
        batch_size = 100
        features = 50
        dropout_rate = 0.5
        
        # Create layer
        layer = Dropout(p=dropout_rate)
        
        # Create input
        x = Tensor(np.ones((batch_size, features)))
        
        # Test training mode
        layer.train()
        output_train = layer(x)
        
        # Check if appropriate number of elements are zeroed out
        zeros_ratio = np.mean(output_train.numpy() == 0)
        self.assertAlmostEqual(zeros_ratio, dropout_rate, delta=0.1)
        
        # Check if non-zero elements are scaled correctly
        nonzero_elements = output_train.numpy()[output_train.numpy() != 0]
        self.assertTrue(np.allclose(nonzero_elements, 1.0 / (1 - dropout_rate)))
        
        # Test evaluation mode
        layer.eval()
        output_eval = layer(x)
        
        # Check if no dropout is applied during evaluation
        self.assertTrue(np.array_equal(output_eval.numpy(), x.numpy()))

if __name__ == '__main__':
    unittest.main()