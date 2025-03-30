# JAX DataLoader

A high-performance data loading library for JAX, designed for efficient data loading and preprocessing in machine learning workflows.

## Features

- Efficient data loading with automatic batching
- Multi-GPU support with automatic batch distribution
- Memory management with automatic batch size tuning
- Support for various data formats (CSV, JSON, Images)
- Progress tracking and statistics
- Data caching and prefetching
- Error handling and recovery

## Installation

```bash
pip install jax-dataloader
```

## Quick Start

```python
from jax_dataloader import JAXDataLoader, DataLoaderConfig

# Create a DataLoader configuration
config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,
    multi_gpu=True
)

# Load your data
dataloader = JAXDataLoader(
    data_path="path/to/your/data",
    config=config
)

# Iterate over batches
for batch_x, batch_y in dataloader:
    # Process your batch
    ...
```

## Examples

The package includes comprehensive examples demonstrating various features:

```bash
# Clone the repository
git clone https://github.com/yourusername/jax-dataloader.git
cd jax-dataloader

# Install example dependencies
pip install -r examples/requirements.txt

# Run the data loading demo
cd examples/data_loading
python demo.py
```

The examples demonstrate:
- Loading different data formats (CSV, JSON, Images)
- Multi-GPU support
- Memory management
- Progress tracking
- Batch size optimization

For more examples and detailed documentation, visit our [documentation](https://jax-dataloader.readthedocs.io/).

## Documentation

For detailed documentation, including API reference and advanced usage examples, visit our [documentation](https://jax-dataloader.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Project Structure**:

```