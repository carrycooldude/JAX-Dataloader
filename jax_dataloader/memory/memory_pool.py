import jax
import jax.numpy as jnp
import numpy as np
import threading
import psutil
from typing import Dict, List, Optional, Tuple
import gc
import os

class NUMAwareMemoryPool:
    def __init__(self, numa_nodes: Optional[List[int]] = None):
        self._pools: Dict[int, Dict[str, List[jnp.ndarray]]] = {}
        self._lock = threading.Lock()
        self._numa_nodes = numa_nodes or list(range(psutil.cpu_count(logical=False)))
        self._initialize_pools()
        
    def _initialize_pools(self):
        """Initialize memory pools for each NUMA node"""
        for node in self._numa_nodes:
            self._pools[node] = {
                'float32': [],
                'float64': [],
                'int32': [],
                'int64': []
            }
    
    def allocate(self, shape: Tuple[int, ...], dtype: str, numa_node: Optional[int] = None) -> jnp.ndarray:
        """Allocate memory with NUMA awareness"""
        if numa_node is None:
            numa_node = self._get_optimal_numa_node()
            
        with self._lock:
            pool = self._pools[numa_node][dtype]
            
            # Try to reuse existing buffer
            for i, buffer in enumerate(pool):
                if buffer.shape == shape:
                    return pool.pop(i)
            
            # Allocate new buffer if none available
            buffer = jnp.zeros(shape, dtype=dtype)
            return buffer
    
    def release(self, buffer: jnp.ndarray, numa_node: Optional[int] = None):
        """Release buffer back to pool"""
        if numa_node is None:
            numa_node = self._get_optimal_numa_node()
            
        with self._lock:
            dtype = str(buffer.dtype)
            self._pools[numa_node][dtype].append(buffer)
    
    def _get_optimal_numa_node(self) -> int:
        """Get optimal NUMA node for current thread"""
        try:
            pid = os.getpid()
            process = psutil.Process(pid)
            cpu_affinity = process.cpu_affinity()
            if cpu_affinity:
                return cpu_affinity[0] // (psutil.cpu_count() // len(self._numa_nodes))
        except:
            pass
        return 0
    
    def cleanup(self):
        """Cleanup unused memory"""
        with self._lock:
            for node_pools in self._pools.values():
                for dtype_pool in node_pools.values():
                    dtype_pool.clear()
            gc.collect()

class MemoryManager:
    def __init__(self):
        self._pool = NUMAwareMemoryPool()
        self._buffer_cache: Dict[str, jnp.ndarray] = {}
        
    def get_buffer(self, key: str, shape: Tuple[int, ...], dtype: str) -> jnp.ndarray:
        """Get or create buffer with caching"""
        if key in self._buffer_cache:
            buffer = self._buffer_cache[key]
            if buffer.shape == shape:
                return buffer
                
        buffer = self._pool.allocate(shape, dtype)
        self._buffer_cache[key] = buffer
        return buffer
    
    def release_buffer(self, key: str):
        """Release buffer from cache"""
        if key in self._buffer_cache:
            self._pool.release(self._buffer_cache.pop(key))
    
    def cleanup(self):
        """Cleanup all buffers"""
        for key in list(self._buffer_cache.keys()):
            self.release_buffer(key)
        self._pool.cleanup() 