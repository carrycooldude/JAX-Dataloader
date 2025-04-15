# jax_dataloader/jax_dataloader.py

import os
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
from typing import Iterator, Union, Optional, Any, Tuple, Dict, List
import numpy as np
import threading
from queue import Queue, Empty, Full
import psutil
import time
import mmap
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import asyncio
import ctypes
import multiprocessing
from jax import profiler
import queue
import warnings
import gc
from .memory.memory_pool import MemoryManager
from .transform.augmentation import JAXDataAugmentation
from .progress.progress_logger import ProgressLogger
from .distributed.distributed_loader import DistributedJAXDataLoader
from transformers import PreTrainedTokenizer
from datasets import Dataset

# Enable maximum performance optimizations
jax.config.update('jax_default_matmul_precision', 'float32')
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_threefry_partitionable', True)
jax.config.update('jax_numpy_rank_promotion', 'allow')
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

class JAXDataLoader:
    def __init__(
        self,
        dataset: Union[Dataset, np.ndarray, jnp.ndarray],
        batch_size: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        shuffle: bool = True,
        num_workers: int = 2,
        prefetch_size: int = 2,
        use_mmap: bool = True,
        use_pinned_memory: bool = True,
        device: Optional[str] = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        augmentations: Optional[List[str]] = None,
        log_file: Optional[str] = None
    ):
        # Initialize memory manager
        self._memory_manager = MemoryManager()
        
        # Setup progress logging
        self._logger = ProgressLogger(log_file=log_file)
        
        # Initialize augmentation
        self._augmenter = JAXDataAugmentation(augmentations)
        
        # Convert dataset to JAX array if needed
        if isinstance(dataset, np.ndarray):
            self.dataset = jnp.array(dataset)
        else:
            self.dataset = dataset
            
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.device = device
        self.distributed = distributed
        
        # Calculate shapes and sizes
        self._dataset_len = len(dataset)
        self._batch_shape = (batch_size,) + dataset.shape[1:]
        self._num_batches = self._dataset_len // batch_size
        
        # Initialize distributed loader if needed
        if distributed:
            self._loader = DistributedJAXDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                tokenizer=tokenizer,
                num_workers=num_workers,
                prefetch_size=prefetch_size,
                device=device,
                world_size=world_size,
                rank=rank
            )
        else:
            # Setup single process loader
            self._batch_queue = Queue(maxsize=prefetch_size)
            self._stop_event = threading.Event()
            self._worker_thread = None
            self._initialize_worker()
            
    def _initialize_worker(self):
        """Initialize worker thread"""
        self._worker_thread = threading.Thread(target=self._worker_fn, daemon=True)
        self._worker_thread.start()
        
    def _worker_fn(self):
        """Worker function for processing batches"""
        indices = np.arange(self._dataset_len)
        while not self._stop_event.is_set():
            if self.shuffle:
                np.random.shuffle(indices)
                
            for i in range(0, self._dataset_len - self.batch_size + 1, self.batch_size):
                if self._stop_event.is_set():
                    break
                    
                start_time = time.time()
                
                # Get batch buffer
                batch_buffer = self._memory_manager.get_buffer(
                    'batch',
                    self._batch_shape,
                    str(self.dataset.dtype)
                )
                
                # Process batch
                batch_indices = indices[i:i + self.batch_size]
                batch = self.dataset[batch_indices]
                
                # Apply augmentations
                batch = self._augmenter.apply(batch)
                
                # Update metrics
                batch_time = time.time() - start_time
                memory_usage = psutil.Process().memory_info().rss
                self._logger.update_batch(batch_time, memory_usage)
                
                # Put batch in queue
                try:
                    self._batch_queue.put(batch, timeout=0.1)
                except Full:
                    continue
                    
    def __iter__(self):
        if self.distributed:
            return iter(self._loader)
        return self
        
    def __next__(self):
        if self.distributed:
            return next(self._loader)
            
        if self._stop_event.is_set():
            raise StopIteration
            
        try:
            batch = self._batch_queue.get(timeout=0.1)
            return batch
        except Empty:
            raise StopIteration
            
    def start_epoch(self, epoch: int, total_epochs: int):
        """Start tracking a new epoch"""
        self._logger.start_epoch(epoch, total_epochs, self._num_batches)
        
    def display_summary(self):
        """Display training summary"""
        self._logger.display_summary()
        
    def save_metrics(self, file_path: str):
        """Save metrics to file"""
        self._logger.save_metrics(file_path)
        
    def __del__(self):
        """Cleanup resources"""
        self._stop_event.set()
        
        # Clear queue
        if not self.distributed:
            while not self._batch_queue.empty():
                try:
                    self._batch_queue.get_nowait()
                except Empty:
                    break
                    
            # Wait for thread
            if self._worker_thread:
                self._worker_thread.join(timeout=0.1)
                
        # Cleanup memory
        self._memory_manager.cleanup()

    def _setup_mmap_storage(self):
        """Setup memory-mapped storage for the dataset"""
        # Create a temporary file for memory mapping
        self._mmap_file = tempfile.NamedTemporaryFile(delete=False)
        self._mmap_file.close()
        
        # Calculate the size needed for the dataset
        dtype_size = np.dtype(self.dataset.dtype).itemsize
        total_size = self.dataset.size * dtype_size
        
        # Check available memory
        available_memory = psutil.virtual_memory().available
        if total_size > available_memory * 0.8:  # Use at most 80% of available memory
            raise MemoryError("Dataset too large for memory mapping")
        
        # Resize the file to the required size
        with open(self._mmap_file.name, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')
        
        # Create memory-mapped array
        self._mmap_array = np.memmap(
            self._mmap_file.name,
            dtype=self.dataset.dtype,
            mode='r+',
            shape=self.dataset.shape
        )
        
        # Copy the dataset to the memory-mapped array
        np.copyto(self._mmap_array, self.dataset, casting='safe')
        
        # Use the memory-mapped array as the dataset
        self.dataset = self._mmap_array
    
    def _setup_pinned_memory(self):
        """Setup pinned memory for faster CPU-GPU transfers"""
        try:
            self._pinned_buffer = np.zeros(
                self._batch_shape,
                dtype=self.dataset.dtype
            )
            # Pin the memory
            ctypes.pythonapi.PyObject_GetBuffer.restype = ctypes.c_int
            ctypes.pythonapi.PyObject_GetBuffer.argtypes = [
                ctypes.py_object,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int
            ]
            buf = ctypes.c_void_p()
            ctypes.pythonapi.PyObject_GetBuffer(
                self._pinned_buffer,
                ctypes.byref(buf),
                0x200  # PyBUF_WRITABLE
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup pinned memory: {e}")