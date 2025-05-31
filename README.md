# Texor - Native Deep Learning Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.0-red.svg)

**Texor** is a lightweight, native deep learning framework built from scratch in Python. It provides a PyTorch-style API without the overhead of large ML frameworks like TensorFlow or PyTorch.

## 🚀 Key Features

- **🎯 Native Implementation**: 100% Python/NumPy/Numba - no TensorFlow or PyTorch dependencies
- **⚡ Lightweight**: ~260MB total size vs ~4GB for TensorFlow + PyTorch
- **🧠 Complete ML Stack**: Automatic differentiation, neural networks, optimizers
- **🔧 PyTorch-style API**: Familiar interface for ML practitioners
- **⚡ JIT Compilation**: Numba-optimized operations for performance
- **🖥️ GPU Ready**: Optional CUDA support via CuPy
- **📦 Easy Installation**: Simple pip install with minimal dependencies

## 🏗️ Architecture

```
texor/
├── core/           # Tensor operations, autograd, backend
├── nn/             # Neural network layers and models
├── optim/          # Optimizers (SGD, Adam, RMSprop)
├── data/           # Dataset utilities and data loaders
└── cli/            # Command-line interface
```

## 📦 Installation

```bash
# Basic installation
pip install numpy numba

# For GPU support (optional)
pip install cupy

# Install Texor
git clone https://github.com/your-repo/texor.git
cd texor
pip install -e .
```

## 🔥 Quick Start

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

## 🧪 Complete Example

```python
from texor.core import randn
from texor.nn import Sequential, Linear, ReLU
from texor.nn.loss import CrossEntropyLoss
from texor.optim import Adam

# Generate sample data
x_train = randn((1000, 20))
y_train = randn((1000, 10))

# Create model
model = Sequential([
    Linear(20, 50),
    ReLU(),
    Linear(50, 10)
])

# Setup training
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()

# Train
for epoch in range(50):
    pred = model(x_train)
    loss = criterion(pred, y_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data.item():.4f}')
```

## 🛠️ Available Components

### Core
- **Tensor**: N-dimensional arrays with automatic differentiation
- **Operations**: Matrix multiplication, element-wise ops, reshaping
- **Backend**: CPU/GPU abstraction with device management

### Neural Networks
- **Layers**: Linear, Conv2D, MaxPool2D, BatchNorm2D, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, ELU, GELU
- **Models**: Sequential container for layer composition

### Loss Functions
- MSELoss, CrossEntropyLoss, BCELoss
- L1Loss, HuberLoss, SmoothL1Loss, KLDivLoss

### Optimizers
- SGD, Adam, RMSprop, AdamW, Adadelta
- Learning rate scheduling and momentum support

## 🎯 Performance Comparison

| Framework | Size | Dependencies | GPU Support | Installation Time |
|-----------|------|--------------|-------------|-------------------|
| **Texor** | ~260MB | 3 packages | ✅ (CuPy) | < 1 min |
| TensorFlow | ~2.1GB | 50+ packages | ✅ | 5-10 min |
| PyTorch | ~1.9GB | 30+ packages | ✅ | 3-7 min |

## 🔧 Command Line Interface

```bash
# Get framework information
python -m texor.cli.main info

# Check dependencies
python -m texor.cli.main check

# List available modules
python -m texor.cli.main list
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_tensor.py -v
python -m pytest tests/test_model.py -v
```

## 🎯 Use Cases

### ✅ Great For:
- **Rapid Prototyping**: Quick ML experiments
- **Educational Projects**: Learning ML algorithms
- **Edge Deployment**: Resource-constrained environments
- **Custom Research**: Need for framework modifications
- **Lightweight Applications**: Minimal dependency requirements

### ⚠️ Consider Alternatives For:
- Large-scale distributed training
- Production systems requiring extensive ecosystem
- Complex pre-trained models from model zoos
- Heavy computer vision pipelines

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by PyTorch's elegant API design
- Built on the shoulders of NumPy and Numba
- Community feedback and contributions

## 📞 Support

- 📧 Email: support@texor.ai
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/texor/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/texor/discussions)

---

**Made with ❤️ for the ML community**
