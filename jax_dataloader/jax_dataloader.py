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
        self._preallocate_batch_memory()
        
        # Initialize thread pool for parallel batch loading
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._batch_queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_signal = object()
        
        # Start prefetching thread
        self._prefetch_thread = threading.Thread(target=self._prefetch_batches)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()

    def _preallocate_batch_memory(self):
        """Pre-allocate memory for batches"""
        sample_shape = self.dataset.shape[1:]
        self._batch_buffer = jnp.zeros((self.batch_size, *sample_shape), dtype=self.dataset.dtype)
        if self.pinned_memory and self.device:
            self._batch_buffer = jax.device_put(self._batch_buffer, device=jax.devices(self.device)[0])

    @staticmethod
    @jax.jit
    def _batch_loader(dataset: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled batch loading function"""
        return jnp.take(dataset, indices, axis=0)

    def _prefetch_batches(self):
        """Prefetch batches in background"""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                break
            
            # Submit batch loading task to thread pool
            future = self._executor.submit(
                self._batch_loader,
                self.dataset,
                jnp.array(batch_indices)
            )
            self._batch_queue.put(future)
        
        self._batch_queue.put(self._stop_signal)

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

    def _python_iterator(self):
        """Iterator using Python backend"""
        while True:
            future = self._batch_queue.get()
            if future is self._stop_signal:
                break
            
            batch = future.result()
            if self.pinned_memory and self.device:
                batch = jax.device_put(batch, device=jax.devices(self.device)[0])
            yield batch

    def __len__(self):
        """Return number of batches"""
        return len(self.dataset) // self.batch_size

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        if hasattr(self, 'rust_loader'):
            del self.rust_loader