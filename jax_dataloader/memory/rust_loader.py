import os
import ctypes
import numpy as np
import jax.numpy as jnp
from typing import Optional, Union, Tuple
import multiprocessing
import psutil

class RustLoader:
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), 'target/release/libjax_loader.so')
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.create_loader.argtypes = [
            ctypes.c_void_p,  # data pointer
            ctypes.c_size_t,  # data size
            ctypes.c_size_t,  # batch size
            ctypes.c_bool,    # shuffle
            ctypes.c_size_t,  # num workers
            ctypes.c_size_t,  # prefetch size
        ]
        self.lib.create_loader.restype = ctypes.c_void_p
        
        self.lib.next_batch.argtypes = [ctypes.c_void_p]
        self.lib.next_batch.restype = ctypes.c_void_p
        
        self.lib.free_loader.argtypes = [ctypes.c_void_p]
        self.lib.free_loader.restype = None
        
        self.loader_ptr = None
        self._data_ref = None  # Keep reference to prevent GC

    def _get_optimal_workers(self) -> int:
        """Get optimal number of worker threads based on system topology"""
        cpu_count = multiprocessing.cpu_count()
        numa_nodes = len(set(psutil.Process().cpu_affinity()))
        # Use 2 workers per NUMA node by default
        return min(cpu_count, numa_nodes * 2)

    def _get_optimal_prefetch(self, batch_size: int) -> int:
        """Get optimal prefetch size based on available memory"""
        mem = psutil.virtual_memory()
        # Target using at most 10% of available memory for prefetching
        target_mem = mem.available * 0.1
        batch_mem = batch_size * self._data_ref.itemsize * np.prod(self._data_ref.shape[1:])
        return max(2, min(16, int(target_mem / batch_mem)))

    def initialize(self, 
                  data: Union[np.ndarray, jnp.ndarray],
                  batch_size: int,
                  shuffle: bool = True,
                  num_workers: Optional[int] = None,
                  prefetch_size: Optional[int] = None):
        """Initialize the Rust loader with data"""
        if isinstance(data, jnp.ndarray):
            data = np.asarray(data)
        
        # Keep reference to prevent garbage collection
        self._data_ref = data
        
        # Get optimal parameters if not specified
        if num_workers is None:
            num_workers = self._get_optimal_workers()
        if prefetch_size is None:
            prefetch_size = self._get_optimal_prefetch(batch_size)
        
        # Ensure data is contiguous and aligned
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Get data pointer and properties
        data_ptr = data.ctypes.data_as(ctypes.c_void_p)
        data_size = data.shape[0]  # Number of samples
        
        self.loader_ptr = self.lib.create_loader(
            data_ptr,
            data_size,
            batch_size,
            shuffle,
            num_workers,
            prefetch_size
        )
        
        self.data_shape = data.shape
        self.dtype = data.dtype
        self.batch_size = batch_size

    def next_batch(self) -> Optional[np.ndarray]:
        """Get next batch from Rust loader"""
        if self.loader_ptr is None:
            raise RuntimeError("Loader not initialized")
        
        batch_ptr = self.lib.next_batch(self.loader_ptr)
        if batch_ptr is None:
            return None
            
        # Create numpy array from the returned pointer
        batch_size = self.batch_size
        total_size = batch_size * np.prod(self.data_shape[1:])
        
        # Create array from memory view with proper alignment
        arr = np.ctypeslib.as_array(
            ctypes.cast(batch_ptr, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(total_size,)
        ).reshape((batch_size, *self.data_shape[1:])).copy()
        
        # Free the Rust-allocated memory
        self.lib.free_batch(batch_ptr)
        
        return arr.astype(self.dtype)

    def __del__(self):
        """Cleanup Rust loader"""
        if self.loader_ptr is not None:
            self.lib.free_loader(self.loader_ptr)
            self.loader_ptr = None
        self._data_ref = None  # Allow data to be garbage collected 