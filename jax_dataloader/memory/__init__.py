"""Memory management module for JAX applications."""

from typing import Any, Dict, Optional, Union
import psutil
import numpy as np
import time

class MemoryManager:
    """Manages memory allocation and deallocation."""
    
    def __init__(self, max_memory: Optional[float] = None):
        """Initialize the memory manager.
        
        Args:
            max_memory: Maximum memory to use in bytes
        """
        self.max_memory = max_memory or get_available_memory()
        self._allocated = 0
        self._peak_usage = 0
        self._start_time = time.time()
        
    def allocate(self, size: int) -> bool:
        """Allocate memory.
        
        Args:
            size: Size in bytes to allocate
            
        Returns:
            True if allocation was successful
        """
        if self._allocated + size > self.max_memory:
            return False
        self._allocated += size
        self._peak_usage = max(self._peak_usage, self._allocated)
        return True
        
    def deallocate(self, size: int):
        """Deallocate memory.
        
        Args:
            size: Size in bytes to deallocate
        """
        self._allocated = max(0, self._allocated - size)
        
    def free(self):
        """Free all allocated memory."""
        self._allocated = 0
        
    def get_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        return {
            "allocated": self._allocated,
            "peak_usage": self._peak_usage,
            "available": self.max_memory - self._allocated,
            "total": self.max_memory,
        }
        
    def cleanup(self):
        """Clean up memory and reset statistics."""
        self.free()
        self._peak_usage = 0
        self._start_time = time.time()
        
    def monitor(self, interval: float = 1.0) -> Dict[str, Any]:
        """Monitor memory usage over time.
        
        Args:
            interval: Time interval between measurements in seconds
            
        Returns:
            Dictionary containing memory monitoring statistics
        """
        current_time = time.time()
        runtime = current_time - self._start_time
        return {
            "runtime": runtime,
            "current_usage": self.get_usage(),
            "memory_per_second": self._peak_usage / runtime if runtime > 0 else 0,
        }
        
    @property
    def allocated(self) -> int:
        """Get the amount of allocated memory."""
        return self._allocated

class Cache:
    """Cache for storing data in memory."""
    
    def __init__(self, max_size: Optional[int] = None):
        """Initialize the cache.
        
        Args:
            max_size: Maximum size of the cache in bytes
        """
        self.max_size = max_size or get_available_memory()
        self._data: Dict[str, Any] = {}
        self._sizes: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if it exists
        """
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None
        
    def put(self, key: str, value: Any, size: Optional[int] = None):
        """Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            size: Size of the value in bytes
        """
        if size is None:
            size = self._estimate_size(value)
            
        while self._total_size + size > self.max_size:
            self._evict()
            
        self._data[key] = value
        self._sizes[key] = size
        
    def clear(self):
        """Clear all data from the cache."""
        self._data.clear()
        self._sizes.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            "size": self._total_size,
            "items": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
        }
        
    def evict(self, key: str):
        """Evict a specific key from the cache.
        
        Args:
            key: Key to evict
        """
        if key in self._data:
            del self._data[key]
            del self._sizes[key]
            self._evictions += 1
        
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        return len(str(value))
        
    def _evict(self):
        """Evict the oldest item from the cache."""
        if not self._data:
            return
        key = next(iter(self._data))
        self.evict(key)
        
    @property
    def _total_size(self) -> int:
        """Get the total size of the cache."""
        return sum(self._sizes.values())

def get_available_memory() -> float:
    """Get the available memory in bytes.
    
    Returns:
        Available memory in bytes
    """
    return psutil.virtual_memory().available
