# JAX DataLoader Documentation

This directory contains the source files for the JAX DataLoader documentation.

## Building the Documentation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

2. Install the package in development mode:

```bash
pip install -e ..
```

### Building

To build the documentation:

```bash
make html
```

The built documentation will be available in `_build/html/`.

### Development

For development with live reload:

```bash
make live
```

This will start a local server that automatically rebuilds the documentation when you make changes.

### Cleaning

To clean the build directory:

```bash
make clean
```

## Documentation Structure

- `index.rst`: Main documentation page
- `installation.rst`: Installation guide
- `usage.rst`: Usage guide with examples
- `api.rst`: API reference
- `examples.rst`: Detailed examples
- `changelog.rst`: Version history and changes

## Contributing

When adding new documentation:

1. Create a new `.rst` file in the appropriate section
2. Add the file to the table of contents in `index.rst`
3. Build the documentation to check for errors
4. Submit a pull request

## Style Guide

- Use reStructuredText syntax
- Follow PEP 8 for Python code examples
- Include docstrings for all public APIs
- Add examples for new features
- Update the changelog for significant changes

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. 