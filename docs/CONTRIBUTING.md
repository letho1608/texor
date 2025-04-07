# Contributing to Texor

We love your input! We want to make contributing to Texor as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nexor.git
cd nexor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- We use `black` for Python code formatting
- We use `isort` for import sorting
- We use `flake8` for style guide enforcement
- We use `mypy` for static type checking

To check your code:
```bash
# Format code
black .
isort .

# Check style
flake8 .

# Type checking
mypy .
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=texor tests/

# Run specific test file
pytest tests/test_tensor.py
```

## Documentation

We use Sphinx for documentation. To build the docs:

```bash
cd docs
make html
```

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the version number in `setup.py` following [SemVer](http://semver.org/)
3. The PR will be merged once you have the sign-off of at least one maintainer

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
