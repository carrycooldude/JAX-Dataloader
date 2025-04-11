"""JAX DataLoader - A high-performance data loading library for JAX applications."""

from .jax_dataloader import JAXDataLoader
from .data import BaseLoader, CSVLoader, JSONLoader, ImageLoader
from .memory import MemoryManager, Cache
from .progress import ProgressTracker
from .transform import Transform
from .exceptions import DataLoaderError, ConfigurationError, MemoryError
from .utils import (
    get_available_memory,
    calculate_batch_size,
    get_device_count,
    format_size
)

__version__ = '0.1.7'
__all__ = [
    'DataLoader',
    'DataLoaderConfig',
    'BaseLoader',
    'CSVLoader',
    'JSONLoader',
    'ImageLoader',
    'MemoryManager',
    'Cache',
    'ProgressTracker',
    'Transform',
    'DataLoaderError',
    'ConfigurationError',
    'MemoryError',
    'get_available_memory',
    'calculate_batch_size',
    'get_device_count',
    'format_size'
]