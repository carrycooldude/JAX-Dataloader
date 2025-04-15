# JAX DataLoader

A high-performance data loading library for JAX applications.

[![PyPI version](https://img.shields.io/pypi/v/jax-dataloaders.svg)](https://pypi.org/project/jax-dataloaders/)  
[![Documentation Status](https://readthedocs.org/projects/jax-dataloader/badge/?version=latest)](https://carrycooldude.github.io/JAX-Dataloader/)  
[![Tests](https://github.com/carrycooldude/JAX-Dataloader/actions/workflows/tests.yml/badge.svg)](https://github.com/carrycooldude/JAX-Dataloader/actions/workflows/tests.yml)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📦 **PyPI**: [jax-dataloaders](https://pypi.org/project/jax-dataloaders/)  
📘 **Documentation**: [https://carrycooldude.github.io/JAX-Dataloader/](https://carrycooldude.github.io/JAX-Dataloader/)  
🌐 **Live Website**: [https://jax-dataloader.netlify.app/](https://jax-dataloader.netlify.app/)  
🔗 **GitHub Repo**: [https://github.com/carrycooldude/JAX-Dataloader](https://github.com/carrycooldude/JAX-Dataloader)

---

## Features

- 🚀 **High Performance**: 1.22x faster than NumPy baseline
- 💾 **Memory Efficient**: Smart memory management with mmap and pinned memory
- 🔄 **Data Augmentation**: JIT-compiled transformations
- 🤗 **HuggingFace Integration**: Native support for datasets and tokenizers
- 📊 **Progress Tracking**: Real-time progress monitoring
- 🔄 **Caching**: Advanced caching strategies
- 🎮 **Multi-GPU Support**: Distributed training ready
- 📈 **Memory Monitoring**: Auto-tuning and optimization

---

## Benchmarks

CPU Performance (100K samples, 512 features, batch size 256):
- JAX DataLoader: 0.1499s per epoch (1.22x faster than NumPy)
- PyTorch DataLoader: 6.2639s per epoch
- TensorFlow DataLoader: 1.6842s per epoch
- NumPy baseline: 0.1829s per epoch

---

## Installation

```bash
pip install jax-dataloaders
```

---

## Quick Start

### Basic Usage

```python
from jax_dataloader import JAXDataLoader
import jax.numpy as jnp

# Create dataset
data = jnp.random.randn(1000, 512)
labels = jnp.random.randint(0, 10, (1000,))

# Create data loader
dataloader = JAXDataLoader(
    dataset=data,
    batch_size=32,
    shuffle=True
)

# Iterate over batches
for batch_data, batch_labels in dataloader:
    print(f"Batch shape: {batch_data.shape}")
```

### HuggingFace Integration

```python
from jax_dataloader import JAXDataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("glue", "sst2", split="train[:1000]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create data loader
dataloader = JAXDataLoader(
    dataset=dataset,
    batch_size=32,
    tokenizer=tokenizer,
    shuffle=True
)

# Iterate over batches
for batch in dataloader:
    print(f"Input IDs shape: {batch['input_ids'].shape}")
```

### Data Augmentation

```python
from jax_dataloader import JAXDataLoader
from jax_dataloader.transform import JAXDataAugmentation

# Create augmenter
augmenter = JAXDataAugmentation(
    augmentations=['random_flip', 'random_rotation', 'color_jitter']
)

# Create data loader with augmentation
dataloader = JAXDataLoader(
    dataset=data,
    batch_size=32,
    augmenter=augmenter
)
```

---

## Documentation

📘 Full documentation available at:  
[https://carrycooldude.github.io/JAX-Dataloader/](https://carrycooldude.github.io/JAX-Dataloader/)

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

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

### Development Tools

We provide several tools to enhance your development experience:

1. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

2. **Development Server**
   ```bash
   python -m jax_dataloader.dev_server
   ```

3. **Benchmarking Tool**
   ```bash
   python -m jax_dataloader.benchmark --help
   ```

4. **Memory Profiler**
   ```bash
   python -m jax_dataloader.profile --help
   ```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=jax_dataloader

# Run specific test file
pytest tests/test_specific_feature.py

# Run tests in parallel
pytest -n auto
```

### Building Documentation
```bash
cd docs
make html
```

### Troubleshooting

Common issues and solutions:

1. **Memory Issues**
   ```python
   # Reduce memory usage
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       use_mmap=True,  # Enable memory mapping
       use_pinned_memory=True,  # Enable pinned memory
       num_workers=1,  # Reduce worker count
       prefetch_size=1  # Reduce prefetch size
   )
   ```

   - **Out of Memory (OOM) Errors**
     ```python
     # Solution: Enable memory mapping and reduce batch size
     loader = JAXDataLoader(
         dataset=data,
         batch_size=16,  # Reduce batch size
         use_mmap=True,
         use_pinned_memory=True
     )
     ```

   - **Memory Leaks**
     ```python
     # Solution: Enable memory tracking and cleanup
     from jax_dataloader.memory import MemoryTracker
     
     with MemoryTracker() as tracker:
         loader = JAXDataLoader(
             dataset=data,
             batch_size=32,
             debug=True
         )
         # Your code here
     tracker.print_report()
     ```

2. **Performance Issues**
   ```python
   # Optimize performance
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       num_workers=4,  # Increase workers
       prefetch_size=2,  # Increase prefetch
       use_mmap=True,  # Enable memory mapping
       use_pinned_memory=True  # Enable pinned memory
   )
   ```

   - **Slow Data Loading**
     ```python
     # Solution: Enable parallel loading and caching
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         num_workers=4,
         prefetch_size=2,
         cache_size=1000  # Cache 1000 batches
     )
     ```

   - **High CPU Usage**
     ```python
     # Solution: Optimize worker count and batch size
     loader = JAXDataLoader(
         dataset=data,
         batch_size=64,  # Increase batch size
         num_workers=2,  # Reduce worker count
         use_mmap=True
     )
     ```

3. **CUDA/GPU Issues**
   ```python
   # Force CPU mode
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   ```

   - **GPU Memory Errors**
     ```python
     # Solution: Enable memory optimization
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         use_mmap=True,
         use_pinned_memory=True,
         gpu_memory_fraction=0.8  # Limit GPU memory usage
     )
     ```

   - **CUDA Device Errors**
     ```python
     # Solution: Check GPU availability and set device
     import jax
     from jax_dataloader.utils import get_available_devices
     
     devices = get_available_devices()
     if not devices:
         jax.config.update('jax_platform_name', 'cpu')
     else:
         loader = JAXDataLoader(
             dataset=data,
             batch_size=32,
             device=devices[0]  # Use first available device
         )
     ```

4. **Data Loading Issues**
   ```python
   # Enable debug mode
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       debug=True  # Enable debug logging
   )
   ```

   - **Batch Shape Mismatch**
     ```python
     # Solution: Verify dataset and batch shapes
     from jax_dataloader.utils import validate_shapes
     
     validate_shapes(dataset, batch_size=32)
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32
     )
     ```

   - **Data Type Errors**
     ```python
     # Solution: Enable type checking and conversion
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         dtype=jnp.float32,  # Specify data type
         convert_types=True  # Enable automatic type conversion
     )
     ```

5. **Distributed Training Issues**
   ```python
   # Solution: Configure distributed settings
   from jax_dataloader.distributed import DistributedConfig
   
   config = DistributedConfig(
       num_nodes=2,
       node_rank=0,
       num_workers=4
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       distributed_config=config
   )
   ```

   - **Synchronization Errors**
     ```python
     # Solution: Enable proper synchronization
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         sync_every_batch=True,  # Synchronize after each batch
         barrier_timeout=30  # Set timeout for synchronization
     )
     ```

   - **Load Balancing Issues**
     ```python
     # Solution: Enable dynamic load balancing
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         dynamic_balancing=True,  # Enable dynamic load balancing
         balance_interval=100  # Rebalance every 100 batches
     )
     ```

6. **Augmentation Issues**
   ```python
   # Solution: Configure augmentation properly
   from jax_dataloader.transform import JAXDataAugmentation
   
   augmenter = JAXDataAugmentation(
       augmentations=['random_flip', 'random_rotation'],
       probability=0.5,  # Set augmentation probability
       seed=42  # Set random seed
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       augmenter=augmenter
   )
   ```

   - **Augmentation Performance**
     ```python
     # Solution: Enable JIT compilation for augmentations
     augmenter = JAXDataAugmentation(
         augmentations=['random_flip', 'random_rotation'],
         jit=True,  # Enable JIT compilation
         parallel=True  # Enable parallel augmentation
     )
     ```

7. **Progress Tracking Issues**
   ```python
   # Solution: Configure progress tracking
   from jax_dataloader.progress import ProgressTracker
   
   tracker = ProgressTracker(
       total_batches=1000,
       update_interval=1,  # Update every batch
       show_eta=True  # Show estimated time remaining
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       progress_tracker=tracker
   )
   ```

   - **Progress Bar Not Updating**
     ```python
     # Solution: Enable proper progress tracking
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         progress_interval=1,  # Update every batch
         show_progress=True,  # Enable progress display
         progress_format='{percentage}% | {bar} | {eta}'  # Custom format
     )
     ```

8. **Caching Issues**
   ```python
   # Solution: Configure caching properly
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       cache_size=1000,  # Cache 1000 batches
       cache_strategy='lru',  # Use LRU caching
       cache_prefetch=True  # Enable cache prefetching
   )
   ```

   - **Cache Memory Usage**
     ```python
     # Solution: Monitor and limit cache usage
     from jax_dataloader.cache import CacheMonitor
     
     monitor = CacheMonitor(
         max_memory_gb=2,  # Limit cache to 2GB
         cleanup_threshold=0.8  # Cleanup at 80% usage
     )
     
     loader = JAXDataLoader(
         dataset=data,
         batch_size=32,
         cache_monitor=monitor
     )
     ```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

Kartikey Rawat

---

## Project Links

- 🔗 GitHub Repo: [https://github.com/carrycooldude/JAX-Dataloader](https://github.com/carrycooldude/JAX-Dataloader)
- 📦 PyPI: [https://pypi.org/project/jax-dataloaders/](https://pypi.org/project/jax-dataloaders/)
- 📘 Docs: [https://carrycooldude.github.io/JAX-Dataloader/](https://carrycooldude.github.io/JAX-Dataloader/)
- 🌐 Website: [https://jax-dataloader.netlify.app/](https://jax-dataloader.netlify.app/)

---
