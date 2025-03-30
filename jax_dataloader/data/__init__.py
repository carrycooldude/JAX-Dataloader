"""Data loading module for JAX applications."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
from jax import random

class BaseLoader(ABC):
    """Base class for all data loaders."""
    
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the data loader.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._rng = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches."""
        pass
        
    @abstractmethod
    def __iter__(self):
        """Return an iterator over the data."""
        pass
        
    def __next__(self):
        """Get the next batch of data."""
        pass

class JSONLoader(BaseLoader):
    """Loader for JSON data."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the JSON loader.
        
        Args:
            data_path: Path to the JSON file
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__(batch_size, shuffle, seed, num_workers, prefetch_factor)
        self.data_path = data_path
        # TODO: Implement JSON loading logic

class ImageLoader(BaseLoader):
    """Loader for image data."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the image loader.
        
        Args:
            data_path: Path to the image directory
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__(batch_size, shuffle, seed, num_workers, prefetch_factor)
        self.data_path = data_path
        # TODO: Implement image loading logic

def get_device_count() -> int:
    """Get the number of available devices.
    
    Returns:
        Number of available devices
    """
    # TODO: Implement device counting logic
    return 1
