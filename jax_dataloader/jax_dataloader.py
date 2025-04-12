# jax_dataloader/jax_dataloader.py

import numpy as np
import jax
import jax.numpy as jnp
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Union, List, Any, Dict
import time

class JAXDataLoader:
    def __init__(
        self, 
        dataset: Union[np.ndarray, jnp.ndarray],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        prefetch_size: Optional[int] = None,
        pinned_memory: bool = True,
        device: Optional[str] = None,
        use_rust: bool = False,  # Default to Python backend
        optimization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JAX DataLoader with optimized backend.
        
        Args:
            dataset: Input dataset as numpy or JAX array
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker threads (auto-configured if None)
            prefetch_size: Size of prefetch queue (auto-configured if None)
            pinned_memory: Whether to use pinned memory
            device: JAX device to use ('cpu', 'gpu', or None for auto)
            use_rust: Whether to use optimized Rust backend
            optimization_config: Additional optimization parameters
        """
        self.dataset = jnp.asarray(dataset) if isinstance(dataset, np.ndarray) else dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers or 4
        self.prefetch_size = prefetch_size or 4
        self.pinned_memory = pinned_memory
        self.device = device
        self.use_rust = use_rust
        self.optimization_config = optimization_config or {}

        # Initialize backend
        if self.use_rust:
            try:
                from .memory.rust_loader import RustLoader
                self.rust_loader = RustLoader()
                self.rust_loader.initialize(
                    self.dataset,
                    self.batch_size,
                    self.shuffle,
                    self.num_workers,
                    self.prefetch_size
                )
            except ImportError:
                print("Warning: Rust backend not available, falling back to Python backend")
                self.use_rust = False
                self._init_python_backend()
        else:
            self._init_python_backend()

    def _init_python_backend(self):
        """Initialize Python backend with optimizations"""
        # Pre-shuffle and pre-compute all batch indices
        self._indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self._indices)
        
        # Pre-compute all batch indices for the entire dataset
        self._batch_indices = np.array([
            self._indices[i:i + self.batch_size]
            for i in range(0, len(self._indices), self.batch_size)
            if len(self._indices[i:i + self.batch_size]) == self.batch_size
        ])
        
        # Pre-allocate memory for all batches
        self._batch_buffer = jnp.zeros(
            (len(self._batch_indices), self.batch_size, *self.dataset.shape[1:]),
            dtype=self.dataset.dtype
        )
        
        if self.pinned_memory and self.device:
            self._batch_buffer = jax.device_put(self._batch_buffer, device=jax.devices(self.device)[0])
        
        # Pre-load all batches using JIT-compiled function
        self._batches = self._batch_loader(self.dataset, jnp.array(self._batch_indices))
        self._current_idx = 0

    @staticmethod
    @jax.jit
    def _batch_loader(dataset: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled batch loading function with vectorized operations"""
        # Reshape indices for vectorized indexing
        batch_size = indices.shape[1]
        indices = indices.reshape(-1)
        # Use advanced indexing for faster access
        batches = dataset[indices]
        # Reshape back to original batch structure
        return batches.reshape(-1, batch_size, *dataset.shape[1:])

    def _python_iterator(self):
        """Iterator using Python backend with pre-loaded batches"""
        while self._current_idx < len(self._batches):
            batch = self._batches[self._current_idx]
            if self.pinned_memory and self.device:
                batch = jax.device_put(batch, device=jax.devices(self.device)[0])
            self._current_idx += 1
            yield batch

    def __iter__(self):
        """Return iterator based on backend"""
        if self.use_rust:
            return self._rust_iterator()
        else:
            return self._python_iterator()

    def _rust_iterator(self):
        """Iterator using Rust backend"""
        while True:
            batch = self.rust_loader.next_batch()
            if batch is None:
                break
            if self.pinned_memory:
                batch = jax.device_put(batch, device=self.device)
            yield batch

    def __len__(self):
        """Return number of batches"""
        return len(self._batch_indices)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'rust_loader'):
            del self.rust_loader