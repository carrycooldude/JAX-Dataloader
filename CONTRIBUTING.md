# Contributing to JAX DataLoader

Thank you for your interest in contributing to JAX DataLoader! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Development Tools](#development-tools)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. **Report Bugs**: Open an issue with the "bug" label
2. **Suggest Features**: Open an issue with the "enhancement" label
3. **Fix Bugs**: Submit a pull request with the "bug" label
4. **Implement Features**: Submit a pull request with the "enhancement" label
5. **Improve Documentation**: Submit a pull request with the "documentation" label

## Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/carrycooldude/JAX-Dataloader.git
   cd JAX-Dataloader
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Development Tools

We provide several tools to make development easier:

1. **Development Server**
   ```bash
   python -m jax_dataloader.dev_server
   ```
   - Hot reloading for development
   - Interactive debugging
   - Memory profiling

2. **Benchmarking Tool**
   ```bash
   python -m jax_dataloader.benchmark --help
   ```
   - Compare performance with other frameworks
   - Profile memory usage
   - Generate benchmark reports

3. **Code Generation**
   ```bash
   python -m jax_dataloader.generate --help
   ```
   - Generate boilerplate code
   - Create test templates
   - Generate documentation stubs

4. **Debugging Tools**
   ```python
   from jax_dataloader.debug import Debugger
   
   # Enable debug mode
   debugger = Debugger()
   debugger.enable()
   
   # Your code here
   
   # Get debug report
   debugger.print_report()
   ```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Additional linting
- **bandit**: Security checking

Run all checks:
```bash
pre-commit run --all-files
```

## Testing

1. **Run Tests**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=jax_dataloader
   
   # Run specific test
   pytest tests/test_specific_feature.py
   
   # Run in parallel
   pytest -n auto
   ```

2. **Test Coverage**
   ```bash
   # Generate coverage report
   pytest --cov=jax_dataloader --cov-report=html
   
   # Open coverage report
   open htmlcov/index.html
   ```

3. **Performance Testing**
   ```bash
   # Run performance tests
   pytest tests/performance/
   
   # Generate performance report
   pytest tests/performance/ --benchmark-only
   ```

## Documentation

1. **Build Documentation**
   ```bash
   cd docs
   make html
   ```

2. **View Documentation**
   ```bash
   open _build/html/index.html
   ```

3. **Documentation Tools**
   ```bash
   # Check documentation coverage
   python -m jax_dataloader.docs --coverage
   
   # Validate docstrings
   python -m jax_dataloader.docs --validate
   ```

## Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests for new features
   - Update documentation
   - Add changelog entry
   - Update type hints
   - Add docstrings

3. **Commit Changes**
   ```bash
   # Use conventional commits
   git commit -m "feat: add new feature"
   git commit -m "fix: resolve bug"
   git commit -m "docs: update documentation"
   ```

4. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Fill out the PR template
   - Link related issues
   - Request reviews
   - Ensure CI passes

## Release Process

1. **Update Version**
   - Update version in `jax_dataloader/__init__.py`
   - Update changelog
   - Update documentation

2. **Create Release**
   ```bash
   # Create and push tag
   git tag vX.Y.Z
   git push origin vX.Y.Z
   
   # Create GitHub release
   gh release create vX.Y.Z
   ```

3. **Build and Upload**
   ```bash
   # Build package
   python -m build
   
   # Upload to PyPI
   twine upload dist/*
   
   # Verify installation
   pip install jax-dataloaders==X.Y.Z
   ```

## Questions?

Feel free to:
- Open an issue
- Join our Discord server
- Contact the maintainers
- Check our FAQ 