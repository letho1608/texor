# Nexor - Comprehensive AI Framework

Nexor is a comprehensive AI framework that combines the best features of TensorFlow and PyTorch. It provides a high-level API while maintaining flexibility and performance through a hybrid backend system.

## Key Features

### 1. Hybrid Backend
- Leverage the power of both TensorFlow and PyTorch
- Seamlessly switch between backends
- Automatic optimization based on use case

### 2. Core API
```python
from nexor.core import Tensor

# Create tensors from various sources
x = Tensor([[1, 2], [3, 4]])  # From Python list
x = Tensor(numpy_array)        # From NumPy array
x = Tensor(tf_tensor)         # From TensorFlow tensor
x = Tensor(torch_tensor)      # From PyTorch tensor

# Access data in multiple formats
numpy_data = x.numpy()
tf_data = x.tensorflow()
torch_data = x.pytorch()
```

### 3. Neural Network Layers
```python
from nexor.nn import Sequential, Linear, Conv2D, MaxPool2D, ReLU, Dropout

model = Sequential([
    Conv2D(in_channels=1, out_channels=32, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Conv2D(in_channels=32, out_channels=64, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Linear(in_features=1600, out_features=10)
])
```

### 4. Optimizers
```python
from nexor.optim import SGD, Adam, RMSprop

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 5. Loss Functions
```python
from nexor.nn import MSELoss, CrossEntropyLoss, BCELoss

# Use loss functions
criterion = CrossEntropyLoss()
loss = criterion(predictions, targets)
```

## Installation

```bash
pip install nexor
```

## Command Line Interface (CLI)

Nexor provides a powerful CLI with intuitive features:

```bash
# View environment and setup information
nexor info

# List available modules
nexor list

# Search for specific modules
nexor list resnet

# Check environment and dependencies
nexor check
```

### CLI Features:
- **Color Output**: Messages, warnings, and errors with clear color coding
- **Progress Bars**: Visual progress for long-running tasks
- **Interactive Interface**: User-friendly command line operations
- **System Information**: Detailed environment and configuration details

## Basic Example

```python
from nexor.nn import Sequential, Linear, ReLU
from nexor.core import Tensor
import numpy as np

# Create model
model = Sequential([
    Linear(input_size=784, output_size=256),
    ReLU(),
    Linear(input_size=256, output_size=10)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

# Create sample data
x = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))
y = np.eye(10)[y]  # One-hot encode

# Train model
model.fit(
    x=Tensor(x),
    y=Tensor(y),
    epochs=10,
    batch_size=32
)
```

## MNIST Example
See `examples/mnist_example.py` for a complete example of training a CNN on the MNIST dataset.

## API Documentation

### Core Module
- `Tensor`: Basic class for tensor operations
- `zeros`, `ones`, `randn`: Tensor creation functions
- `from_numpy`, `from_tensorflow`, `from_pytorch`: Conversions from other formats

### Neural Network (nn) Module
- Layers: `Linear`, `Conv2D`, `MaxPool2D`, `Dropout`
- Activations: `ReLU`, `Sigmoid`, `Tanh`
- Loss Functions: `MSELoss`, `CrossEntropyLoss`, `BCELoss`
- Model: `Sequential` - Easy-to-use API for model building

### Optimizers Module
- `SGD`: Stochastic Gradient Descent with momentum
- `Adam`: Adam optimizer
- `RMSprop`: RMSprop optimizer

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for more details.

## License

MIT License - see the `LICENSE` file for details.

## Language Support

For Vietnamese documentation, please see [README_VN.md](docs/README_VN.md).