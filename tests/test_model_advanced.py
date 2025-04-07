import unittest
import numpy as np
import tensorflow as tf
import torch
from nexor.core import Tensor
from nexor.nn import Sequential, Linear, ReLU, Conv2D, MaxPool2D
from nexor.nn.advanced_layers import ResidualBlock, LSTM
from nexor.optim import Adam
from nexor.core.backend import backend

class TestModelAdvanced(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        backend.set_backend('auto')
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)

    def test_complex_model_training(self):
        """Test training of a complex model with multiple backends"""
        # Create a CNN with residual connections
        model = Sequential([
            Conv2D(3, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64),
            MaxPool2D(kernel_size=2),
            ResidualBlock(64, 128, stride=2),
            MaxPool2D(kernel_size=2),
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            Linear(128 * 8 * 8, 10)
        ])

        # Create sample data
        x = Tensor(np.random.randn(32, 3, 32, 32))
        y = Tensor(np.eye(10)[np.random.randint(0, 10, 32)])

        # Test with different backends
        backends = ['tensorflow', 'pytorch']
        for backend_name in backends:
            backend.set_backend(backend_name)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Train
            history = model.fit(
                x=x,
                y=y,
                epochs=3,
                batch_size=8,
                validation_split=0.2
            )

            # Check training metrics
            self.assertIn('loss', history)
            self.assertIn('accuracy', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_accuracy', history)

    def test_model_save_load(self):
        """Test model saving and loading across backends"""
        model = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 1)
        ])

        x = Tensor(np.random.randn(10, 10))
        original_output = model(x)

        # Save model
        model.save('test_model.h5')

        # Load model with different backend
        backend.set_backend('tensorflow' if backend.current == 'pytorch' else 'pytorch')
        loaded_model = Sequential.load('test_model.h5')

        # Check outputs are the same
        loaded_output = loaded_model(x)
        self.assertTrue(np.allclose(original_output.numpy(), 
                                  loaded_output.numpy(), rtol=1e-5))

    def test_mixed_precision_training(self):
        """Test mixed precision training"""
        model = Sequential([
            Linear(100, 50),
            ReLU(),
            Linear(50, 10)
        ])

        x = Tensor(np.random.randn(32, 100))
        y = Tensor(np.eye(10)[np.random.randint(0, 10, 32)])

        # Enable mixed precision
        backend.enable_mixed_precision()

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            x=x,
            y=y,
            epochs=3,
            batch_size=8
        )

        # Check if training completed successfully
        self.assertIn('loss', history)
        self.assertTrue(all(not np.isnan(loss) for loss in history['loss']))

    def test_multi_gpu_training(self):
        """Test multi-GPU training if available"""
        if not torch.cuda.device_count() > 1:
            self.skipTest("Multiple GPUs not available")

        model = Sequential([
            Conv2D(3, 64, kernel_size=3),
            ResidualBlock(64, 64),
            MaxPool2D(kernel_size=2),
            Linear(64 * 14 * 14, 10)
        ])

        x = Tensor(np.random.randn(64, 3, 32, 32))
        y = Tensor(np.eye(10)[np.random.randint(0, 10, 64)])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train with data parallel
        history = model.fit(
            x=x,
            y=y,
            epochs=3,
            batch_size=16,
            use_multi_gpu=True
        )

        self.assertIn('loss', history)

    def test_gradient_accumulation(self):
        """Test gradient accumulation for large models"""
        model = Sequential([
            Linear(1000, 500),
            ReLU(),
            Linear(500, 100)
        ])

        x = Tensor(np.random.randn(128, 1000))
        y = Tensor(np.random.randn(128, 100))

        model.compile(
            optimizer='adam',
            loss='mse'
        )

        # Train with gradient accumulation
        history = model.fit(
            x=x,
            y=y,
            epochs=3,
            batch_size=16,
            accumulation_steps=4
        )

        self.assertIn('loss', history)

    def test_training_callbacks(self):
        """Test training callbacks"""
        class TestCallback:
            def __init__(self):
                self.called = {
                    'on_epoch_begin': 0,
                    'on_epoch_end': 0,
                    'on_batch_begin': 0,
                    'on_batch_end': 0
                }

            def on_epoch_begin(self, epoch):
                self.called['on_epoch_begin'] += 1

            def on_epoch_end(self, epoch, logs):
                self.called['on_epoch_end'] += 1

            def on_batch_begin(self, batch):
                self.called['on_batch_begin'] += 1

            def on_batch_end(self, batch, logs):
                self.called['on_batch_end'] += 1

        model = Sequential([
            Linear(10, 1)
        ])

        x = Tensor(np.random.randn(32, 10))
        y = Tensor(np.random.randn(32, 1))

        callback = TestCallback()
        model.compile(optimizer='adam', loss='mse')
        model.fit(
            x=x,
            y=y,
            epochs=2,
            batch_size=8,
            callbacks=[callback]
        )

        # Check if all callbacks were called
        self.assertEqual(callback.called['on_epoch_begin'], 2)
        self.assertEqual(callback.called['on_epoch_end'], 2)
        self.assertEqual(callback.called['on_batch_begin'], 8)  # 32/8 * 2
        self.assertEqual(callback.called['on_batch_end'], 8)

if __name__ == '__main__':
    unittest.main()