# Texor - Native Deep Learning Framework

**Texor** is a lightweight, native deep learning framework built from scratch in Python. It provides a PyTorch-style API without the overhead of large ML frameworks like TensorFlow or PyTorch.

## Key Features

- Native Implementation: 100% Python/NumPy/Numba - no TensorFlow or PyTorch dependencies
- Lightweight: ~260MB total size vs ~4GB for TensorFlow + PyTorch
- Complete ML Stack: Automatic differentiation, neural networks, optimizers
- PyTorch-style API: Familiar interface for ML practitioners
- JIT Compilation: Numba-optimized operations for performance
- GPU Ready: Optional CUDA support via CuPy
- Easy Installation: Simple pip install with minimal dependencies

## Recent Updates

### Version 1.0.0 (Latest)
-  **32/32 tests passing** - Complete test coverage
-  **Enhanced activation functions** - Added Tanh, improved Sigmoid
-  **Optimized loss functions** - Better gradient flow and numerical stability
-  **Improved optimizers** - Enhanced convergence properties
-  **Enhanced CLI** - Better user experience and error handling
-  **Performance optimizations** - Faster tensor operations and memory management

## Architecture

```
texor/
├── core/           # Tensor operations, autograd, backend
├── nn/             # Neural network layers and models
├── optim/          # Optimizers (SGD, Adam, RMSprop)
├── data/           # Dataset utilities and data loaders
└── cli/            # Command-line interface
```

## Installation

```bash
# Basic installation
pip install numpy numba

# For GPU support (optional)
pip install cupy

# Install Texor
git clone https://github.com/letho1608/texor
cd texor
pip install -e .
```

## Quick Start

### Basic Tensor Operations
```python
import texor
from texor.core import Tensor, randn

# Create tensors
x = randn((3, 4))
y = randn((4, 2))

# Matrix operations with autograd
z = x @ y
z.backward()
print(x.grad)  # Gradients computed automatically
```

### Neural Networks
```python
from texor.nn import Sequential, Linear, ReLU
from texor.nn.loss import MSELoss
from texor.optim import Adam

# Define model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64), 
    ReLU(),
    Linear(64, 10)
])

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### High-level API
```python
# Keras-style high-level API
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## Complete Example

```python
from texor.core import randn
from texor.nn import Sequential, Linear, ReLU
```

## Performance Benchmarks

### Test Results
- **32/32 tests passing** (100% success rate)
- **Memory efficient** operations
- **Fast tensor computations** with Numba JIT
- **Stable gradients** with numerical optimization

### Supported Operations
- ✅ Tensor operations (add, multiply, matrix operations)
- ✅ Automatic differentiation (autograd)
- ✅ Neural network layers (Linear, Conv2D, MaxPool2D)
- ✅ Activation functions (ReLU, Sigmoid, Tanh)
- ✅ Loss functions (MSE, CrossEntropy, BCE, L1, Huber)
- ✅ Optimizers (SGD, Adam, RMSprop, Adagrad, Adadelta)
- ✅ Data utilities (Dataset, DataLoader, Transforms)
- ✅ Model management (Sequential, Model class)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch's elegant API design
- Built with NumPy and Numba for performance
- Thanks to the open-source community for inspiration and tools


Made with care for the ML community
