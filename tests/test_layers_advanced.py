import unittest
import numpy as np
import tensorflow as tf
import torch
from nexor.core import Tensor
from nexor.nn.layers import Linear, Conv2D, MaxPool2D, Dropout
from nexor.core.backend import backend
from nexor.nn.advanced_layers import ResidualBlock, LSTM, SelfAttention, TransformerEncoderLayer

class TestLayersAdvanced(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        backend.set_backend('auto')
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)

    def test_residual_block(self):
        """Test ResidualBlock functionality"""
        batch_size = 16
        in_channels = 64
        out_channels = 128
        height = width = 32
        
        block = ResidualBlock(in_channels, out_channels, stride=2)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        expected_height = height // 2
        expected_width = width // 2
        self.assertEqual(output.shape, 
                        (batch_size, out_channels, expected_height, expected_width))
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed for all parameters
        self.assertIsNotNone(block.conv1.weight.grad)
        self.assertIsNotNone(block.conv2.weight.grad)
        self.assertIsNotNone(block.shortcut.weight.grad)

    def test_lstm_layer(self):
        """Test LSTM layer functionality"""
        batch_size = 32
        seq_length = 10
        input_size = 20
        hidden_size = 50
        
        lstm = LSTM(input_size, hidden_size, num_layers=2, bidirectional=True)
        x = Tensor(np.random.randn(batch_size, seq_length, input_size))
        
        # Forward pass
        output, (h_n, c_n) = lstm(x)
        
        # Check output shapes
        self.assertEqual(output.shape, 
                        (batch_size, seq_length, hidden_size * 2))  # *2 for bidirectional
        self.assertEqual(h_n.shape, 
                        (4, batch_size, hidden_size))  # 4 = 2 layers * 2 directions
        self.assertEqual(c_n.shape, 
                        (4, batch_size, hidden_size))
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed
        self.assertTrue(all(w.grad is not None for w in lstm.w_ih))
        self.assertTrue(all(w.grad is not None for w in lstm.w_hh))

    def test_self_attention(self):
        """Test SelfAttention layer functionality"""
        batch_size = 8
        seq_length = 16
        embed_dim = 64
        num_heads = 4
        
        attention = SelfAttention(embed_dim, num_heads)
        x = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
        
        # Create attention mask
        mask = Tensor(np.triu(np.ones((seq_length, seq_length)), k=1).astype(np.bool_))
        
        # Forward pass
        output = attention(x, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, embed_dim))
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed
        self.assertIsNotNone(attention.q_proj.weight.grad)
        self.assertIsNotNone(attention.k_proj.weight.grad)
        self.assertIsNotNone(attention.v_proj.weight.grad)

    def test_transformer_encoder_layer(self):
        """Test TransformerEncoderLayer functionality"""
        batch_size = 8
        seq_length = 16
        d_model = 64
        nhead = 4
        dim_feedforward = 256
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=0.1
        )
        x = Tensor(np.random.randn(batch_size, seq_length, d_model))
        
        # Forward pass with different backend configurations
        for backend_name in ['tensorflow', 'pytorch']:
            backend.set_backend(backend_name)
            
            # Test with and without attention mask
            mask = None
            output1 = encoder_layer(x, mask)
            self.assertEqual(output1.shape, (batch_size, seq_length, d_model))
            
            mask = Tensor(np.triu(np.ones((seq_length, seq_length)), k=1).astype(np.bool_))
            output2 = encoder_layer(x, mask)
            self.assertEqual(output2.shape, (batch_size, seq_length, d_model))
            
            # Check that outputs are different with and without mask
            self.assertFalse(np.allclose(output1.numpy(), output2.numpy()))

    def test_backend_switching(self):
        """Test layer behavior when switching backends"""
        batch_size = 16
        in_channels = 3
        out_channels = 64
        height = width = 32
        
        conv = Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        # Test with different backends
        backends = ['tensorflow', 'pytorch']
        outputs = []
        
        for backend_name in backends:
            backend.set_backend(backend_name)
            outputs.append(conv(x).numpy())
            
        # Check if outputs are similar across backends
        self.assertTrue(np.allclose(outputs[0], outputs[1], rtol=1e-4))

    def test_memory_leaks(self):
        """Test for memory leaks in layers"""
        import gc
        import sys
        
        def create_and_run_layer():
            layer = Linear(1000, 1000)
            x = Tensor(np.random.randn(100, 1000))
            y = layer(x)
            loss = y.sum()
            loss.backward()
            
        initial_objects = len(gc.get_objects())
        
        for _ in range(10):
            create_and_run_layer()
            gc.collect()
            
        final_objects = len(gc.get_objects())
        
        # Allow for some small increase but catch significant leaks
        self.assertLess(final_objects - initial_objects, 100)

if __name__ == '__main__':
    unittest.main()