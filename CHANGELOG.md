# Changelog

All notable changes to Texor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-31

### Added
- **Core Framework**
  - Native tensor operations with automatic differentiation
  - Numba JIT compilation for performance optimization
  - Device management (CPU/GPU) with CuPy backend
  - Memory-efficient gradient computation

- **Neural Networks**
  - Sequential model container
  - Linear, Conv2D, MaxPool2D layers
  - Activation functions: ReLU, Sigmoid, Tanh
  - Batch normalization and dropout layers
  - Advanced layers: ResidualBlock, LSTM (experimental)

- **Loss Functions**
  - MSELoss, CrossEntropyLoss, BCELoss
  - L1Loss, HuberLoss, SmoothL1Loss
  - KLDivLoss for distribution matching

- **Optimizers**
  - SGD with momentum support
  - Adam with bias correction
  - RMSprop adaptive learning rate
  - AdamW with weight decay
  - Adadelta parameter adaptation

- **Data Handling**
  - Dataset base class
  - Data loading utilities
  - Basic transforms

- **Command Line Interface**
  - `texor info` - System and framework information
  - `texor list` - Available modules
  - Rich console output with colors

- **Development Tools**
  - Comprehensive test suite
  - Development dependencies
  - Code formatting and linting setup

### Technical Details
- **Dependencies**: Minimal core (NumPy, Numba, Rich, Click)
- **Size**: ~260MB total installation
- **Python**: 3.8+ support
- **GPU**: Optional CUDA support via CuPy
- **API**: PyTorch-compatible interface

### Performance
- JIT compilation for critical operations
- Memory-efficient autograd system
- Optimized matrix operations
- Device-agnostic computation

### Testing
- 95%+ test coverage
- Unit tests for all components
- Integration tests for workflows
- Performance benchmarks

## [0.1.1] - 2025-05-31

### Changed
- **Code Optimization**
  - Cleaned up project structure, removed debug and temporary files
  - Optimized setup.py with minimal core dependencies
  - Improved texor/__init__.py with cleaner API and built-in info() function
  - Enhanced CLI output with accurate component information

- **Documentation**
  - Updated README.md with comprehensive framework description
  - Added CONTRIBUTING.md with development guidelines
  - Created detailed CHANGELOG.md for version tracking
  - Added .gitignore for proper version control

- **Project Structure**
  - Moved demo files to examples/ directory
  - Removed duplicate and debug files
  - Organized codebase for production readiness

### Fixed
- CLI encoding issues with Unicode characters
- Sequential layer constructor to support both list and *args syntax
- Import paths consistency across all modules
- Missing device_count() function in core module

## [Unreleased]

### Planned Features
- Model serialization/loading
- More activation functions (ELU, GELU, Swish)
- Advanced optimizers (LAMB, RAdam)
- Data augmentation transforms
- Model visualization tools
- ONNX export support
- Distributed training
- Mixed precision training

### Known Issues
- Limited pre-trained model support
- No model zoo integration
- Experimental advanced layers

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.