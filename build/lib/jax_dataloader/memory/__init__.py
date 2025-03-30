"""Memory management module for JAX applications."""

from typing import Any, Dict, Optional, Union
import psutil
import numpy as np

class MemoryManager:
    """Manages memory allocation and deallocation."""
    
    def __init__(self, max_memory: Optional[float] = None):
        """Initialize the memory manager.
        
        Args:
            max_memory: Maximum memory to use in bytes
        """
        self.max_memory = max_memory or get_available_memory()
        self._allocated = 0
        
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
        return True
        
    def deallocate(self, size: int):
        """Deallocate memory.
        
        Args:
            size: Size in bytes to deallocate
        """
        self._allocated = max(0, self._allocated - size)
        
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
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if it exists
        """
        return self._data.get(key)
        
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
        del self._data[key]
        del self._sizes[key]
        
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
