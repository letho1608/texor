# Contributing to Texor

Thank you for your interest in contributing to Texor! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy, Numba
- Git

### Development Setup
```bash
# Clone the repository
git clone https://github.com/letho1608/texor.git
cd texor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests to verify setup
python -m pytest tests/ -v
```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and modular

### Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=texor --cov-report=html
```

### Documentation
- Update README.md for new features
- Add docstrings following NumPy style
- Include examples in docstrings

## 🛠️ Types of Contributions

### 🐛 Bug Reports
- Use GitHub Issues
- Include minimal reproduction example
- Specify Python version and OS
- Include error messages and stack traces

### ✨ Feature Requests
- Describe the use case
- Explain why it would be useful
- Consider API design implications

### 🔧 Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Commit with descriptive messages
7. Push to your fork
8. Open a Pull Request

## 📦 Project Structure

```
texor/
├── texor/
│   ├── core/           # Core tensor operations
│   ├── nn/             # Neural network components
│   ├── optim/          # Optimizers
│   ├── data/           # Data handling
│   └── cli/            # Command line interface
├── tests/              # Test suite
├── examples/           # Usage examples
└── tools/              # Development tools
```

## 🧪 Testing Strategy

### Unit Tests
- Test individual functions/classes
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Tests
- Test component interactions
- Verify end-to-end workflows
- Performance benchmarks

## 📝 Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(nn): add dropout layer implementation

fix(core): resolve gradient computation for complex operations

docs(readme): update installation instructions
```

## 🔄 Pull Request Process

1. **Before Creating PR**
   - Sync with main branch
   - Run all tests
   - Update documentation
   - Add entry to CHANGELOG.md

2. **PR Description**
   - Clear title and description
   - Link related issues
   - Include testing information
   - Add screenshots if UI changes

3. **Review Process**
   - Code review by maintainers
   - Automated testing via CI
   - Address feedback promptly
   - Squash commits before merge

## 🏗️ Architecture Decisions

### Core Principles
- **Native Implementation**: Minimize external dependencies
- **Performance**: Optimize for speed and memory
- **Simplicity**: Keep API intuitive and clean
- **Compatibility**: PyTorch-style interface

### Adding New Features
- Consider impact on existing API
- Ensure consistent naming conventions
- Add comprehensive tests
- Update documentation

## 🐞 Debugging

### Common Issues
- Import errors: Check PYTHONPATH
- CUDA errors: Verify CuPy installation
- Performance: Profile with `cProfile`

### Debugging Tools
```python
# Enable debug mode
import texor
texor.set_debug(True)

# Profile operations
import cProfile
cProfile.run('your_code()')
```

## 📊 Performance Guidelines

- Benchmark before and after changes
- Use Numba JIT compilation for hot paths
- Minimize memory allocations
- Consider GPU acceleration opportunities

## 🆘 Getting Help

- 💬 **Discussions**: GitHub Discussions for questions
- 🐛 **Issues**: GitHub Issues for bugs
- 📧 **Email**: letho16082003@gmail.com for security issues

## 📄 License

By contributing to Texor, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Texor better! 🎉