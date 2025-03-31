# JAX DataLoader

A high-performance data loading library for JAX applications.

## Features

- Efficient data loading with memory management
- Support for CSV, JSON, and image data
- Data augmentation and preprocessing
- Progress tracking
- Caching
- Multi-GPU support
- Memory monitoring and auto-tuning

## Installation

```bash
pip install jax-dataloaders
```

## Usage

### Basic Usage

```python
from jax_dataloader import DataLoader, DataLoaderConfig

# Create configuration
config = DataLoaderConfig(
    batch_size=32,
    shuffle=True,
    loader_type="csv",
    data_path="data.csv"
)

# Create data loader
dataloader = DataLoader(config)

# Iterate over batches
for batch_data, batch_labels in dataloader:
    print(f"Batch shape: {batch_data.shape}")
```

### Advanced Usage

```python
from jax_dataloader import DataLoader, DataLoaderConfig
from jax_dataloader.transform import Transform

# Create transform
transform = Transform()
transform.add(lambda x: x * 2)  # Example transform

# Create configuration
config = DataLoaderConfig(
    batch_size=32,
    shuffle=True,
    loader_type="json",
    data_path="data.json",
    transform=transform,
    num_workers=4,
    prefetch_factor=2
)

# Create data loader
dataloader = DataLoader(config)

# Iterate over batches
for batch_data, batch_labels in dataloader:
    print(f"Batch shape: {batch_data.shape}")
```

## Documentation

For detailed documentation, visit [https://carrycooldude.github.io/JAX-Dataloader/](https://carrycooldude.github.io/JAX-Dataloader/).

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/carrycooldude/JAX-Dataloader.git
cd JAX-Dataloader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Testing

Run tests:
```bash
pytest
```

### Building Documentation

Build documentation:
```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Kartikey Rawat

## Repository

GitHub: [https://github.com/carrycooldude/JAX-Dataloader](https://github.com/carrycooldude/JAX-Dataloader)

---

### **Project Structure**:

```
