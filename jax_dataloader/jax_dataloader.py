# jax_dataloader/jax_dataloader.py

import os
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
from typing import Iterator, Union, Optional, Any, Tuple
import numpy as np
import threading
from queue import Queue, Empty, Full
import psutil
import time
import mmap
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

# Enable maximum performance optimizations
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=8'

class JAXDataLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        prefetch_size: int = 4,
        seed: int = 42,
        use_mmap: bool = True
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.use_mmap = use_mmap
        
        # Initialize queues and signals
        self.work_queue = Queue(maxsize=prefetch_size * 2)
        self.batch_queue = Queue(maxsize=prefetch_size)
        self.stop_signal = threading.Event()
        self.workers = []
        
        # Convert dataset to JAX array and optimize memory layout
        if not isinstance(dataset, jnp.ndarray):
            self.dataset = jnp.asarray(dataset, dtype=jnp.float32)
        else:
            self.dataset = dataset
            
        # Pre-allocate memory for batches
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches += 1
            
        # Initialize state
        self.key = random.PRNGKey(seed)
        self.current_batch = 0
        
        # Setup memory-mapped storage if enabled
        if self.use_mmap:
            self._setup_mmap_storage()
        else:
            self.mmap_file = None
            self.mmap_buffer = None
            
        # Setup indices and pre-compile functions
        self._setup_indices()
        self._get_batch = self._compile_batch_fn()
        
        # Pre-allocate memory for batches
        self._batch_shape = (self.batch_size,) + self.dataset.shape[1:]
        self._batch_buffer = jnp.zeros(self._batch_shape)
        
        # Setup worker threads with thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix='jax_dataloader'
        )
        self._setup_workers()
        
    def _setup_mmap_storage(self):
        """Setup memory-mapped storage for efficient data access"""
        # Create temporary file for memory mapping
        self.mmap_file = tempfile.NamedTemporaryFile(delete=False)
        self.mmap_file.close()
        
        # Calculate total size needed
        dtype_size = np.dtype(np.float32).itemsize
        total_size = self.num_samples * self.dataset.shape[1] * dtype_size
        
        # Resize file to required size
        with open(self.mmap_file.name, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')
            
        # Create memory-mapped buffer
        self.mmap_buffer = np.memmap(
            self.mmap_file.name,
            dtype=np.float32,
            mode='r+',
            shape=self.dataset.shape
        )
        
        # Copy data to memory-mapped buffer in chunks
        chunk_size = 100000
        for i in range(0, self.num_samples, chunk_size):
            end = min(i + chunk_size, self.num_samples)
            self.mmap_buffer[i:end] = np.asarray(self.dataset[i:end])
        self.mmap_buffer.flush()
        
        # Convert to JAX array with optimized memory layout
        self.dataset = jnp.asarray(self.mmap_buffer)
        
    def _setup_workers(self):
        """Setup worker threads for parallel batch processing"""
        # Start prefetching
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name='jax_dataloader_prefetch'
        )
        self.prefetch_thread.start()
        
    def _prefetch_loop(self):
        """Prefetch loop to keep batch queue filled"""
        idx = 0
        while idx < self.num_batches and not self.stop_signal.is_set():
            try:
                # Submit batch processing to thread pool
                future = self.thread_pool.submit(
                    self._process_batch,
                    idx
                )
                future.add_done_callback(
                    lambda f: self._handle_batch_result(f, idx)
                )
                idx += 1
            except Full:
                # If work queue is full, wait a bit
                time.sleep(0.01)
            
    def _process_batch(self, idx: int) -> jnp.ndarray:
        """Process a single batch"""
        # Get batch data
        batch = self._get_batch(
            self.dataset,
            self.indices[idx]
        )
        
        # Apply mask
        batch = batch * self._batch_masks[idx]
        
        return batch
        
    def _handle_batch_result(self, future, idx: int):
        """Handle batch processing result"""
        try:
            batch = future.result()
            try:
                self.batch_queue.put(batch, timeout=0.1)
            except Full:
                # If queue is full, skip this batch
                pass
        except Exception:
            # Handle any errors in batch processing
            pass
            
    def _setup_indices(self):
        """Setup indices and masks for efficient batch extraction"""
        # Calculate padding
        remainder = self.num_samples % self.batch_size
        padding_size = 0 if remainder == 0 or self.drop_last else self.batch_size - remainder
        total_size = self.num_samples + padding_size
        
        # Create indices with padding
        self.indices = jnp.arange(total_size)
        self.valid_mask = jnp.where(
            self.indices < self.num_samples,
            jnp.ones(total_size, dtype=jnp.float32),
            jnp.zeros(total_size, dtype=jnp.float32)
        )
        
        # Reshape for batching
        self.indices = self.indices.reshape(-1, self.batch_size)
        self.valid_mask = self.valid_mask.reshape(-1, self.batch_size)
        
        # Pre-compile masks for each batch
        self._batch_masks = [
            self.valid_mask[i][..., None] for i in range(len(self.valid_mask))
        ]
        
    def _compile_batch_fn(self):
        """Compile optimized batch extraction function with vectorization"""
        @jit
        def get_batch(data: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
            # Use vectorized operations for batch extraction
            def get_slice(idx):
                return lax.dynamic_slice(data, (idx,) + (0,) * (data.ndim - 1), (1,) + data.shape[1:])
            return vmap(get_slice)(indices).reshape(self._batch_shape)
        return get_batch
    
    def __iter__(self) -> Iterator[jnp.ndarray]:
        if self.shuffle:
            # Get shuffled indices
            self.key, subkey = random.split(self.key)
            perm = random.permutation(subkey, self.num_samples)
            # Update indices with shuffled values
            num_full_batches = (self.num_samples // self.batch_size) * self.batch_size
            self.indices = self.indices.at[:-1].set(
                perm[:num_full_batches].reshape(-1, self.batch_size)
            )
            if not self.drop_last and self.num_samples % self.batch_size != 0:
                last_batch = jnp.pad(
                    perm[num_full_batches:],
                    (0, self.batch_size - (self.num_samples % self.batch_size)),
                    constant_values=self.num_samples-1
                )
                self.indices = self.indices.at[-1].set(last_batch)
        
        self.current_batch = 0
        return self
    
    def __next__(self) -> jnp.ndarray:
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        try:
            batch = self.batch_queue.get(timeout=0.1)
            self.current_batch += 1
            return batch
        except Empty:
            raise StopIteration
    
    def __len__(self) -> int:
        return self.num_batches
        
    def __del__(self):
        """Cleanup resources"""
        self.stop_signal.set()
        self.thread_pool.shutdown(wait=False)
        self.prefetch_thread.join(timeout=0.1)
        
        # Cleanup memory-mapped file
        if self.mmap_file is not None:
            try:
                self.mmap_buffer = None
                os.unlink(self.mmap_file.name)
            except:
                pass