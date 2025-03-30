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
            max_memory: Maximum memory to allocate in bytes
        """
        self.max_memory = max_memory or psutil.virtual_memory().total
        self.allocated = 0
        self._peak_usage = 0
        self._last_update = time.time()
        
    def allocate(self, size: int) -> bool:
        """Allocate memory.
        
        Args:
            size: Size to allocate in bytes
            
        Returns:
            True if allocation was successful
        """
        if self.allocated + size > self.max_memory:
            return False
            
        self.allocated += size
        self._peak_usage = max(self._peak_usage, self.allocated)
        return True
        
    def deallocate(self, size: int):
        """Deallocate memory.
        
        Args:
            size: Size to deallocate in bytes
        """
        self.allocated = max(0, self.allocated - size)
        
    def free(self, size: Optional[int] = None):
        """Free memory.
        
        Args:
            size: Optional size to free in bytes
        """
        if size is not None:
            self.deallocate(size)
        else:
            self.allocated = 0
            
    def get_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        return {
            "allocated": self.allocated,
            "peak_usage": self._peak_usage,
            "available": self.max_memory - self.allocated,
            "total": self.max_memory
        }
        
    def cleanup(self):
        """Clean up memory and reset statistics."""
        self.allocated = 0
        self._peak_usage = 0
        self._last_update = time.time()
        
    def monitor(self, interval: float = 1.0) -> Dict[str, Any]:
        """Monitor memory usage over time.
        
        Args:
            interval: Time interval between updates in seconds
            
        Returns:
            Dictionary containing runtime statistics
        """
        current_time = time.time()
        if current_time - self._last_update >= interval:
            stats = self.get_usage()
            stats["timestamp"] = current_time
            self._last_update = current_time
            return stats
        return {}

class Cache:
    """Cache for storing data in memory."""
    
    def __init__(
        self,
        max_size: int,
        eviction_policy: str = "lru",
        track_stats: bool = True,
        max_age: Optional[float] = None,
    ):
        """Initialize the cache.
        
        Args:
            max_size: Maximum size of the cache in bytes
            eviction_policy: Cache eviction policy ("lru" or "fifo")
            track_stats: Whether to track cache statistics
            max_age: Maximum age of cached items in seconds
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.track_stats = track_stats
        self.max_age = max_age
        
        self._cache: Dict[str, Any] = {}
        self._sizes: Dict[str, int] = {}
        self._timestamps: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if key in self._cache:
            if self.max_age is not None:
                if time.time() - self._timestamps[key] > self.max_age:
                    self.evict(key)
                    if self.track_stats:
                        self._misses += 1
                    return None
                    
            if self.track_stats:
                self._hits += 1
            return self._cache[key]
            
        if self.track_stats:
            self._misses += 1
        return None
        
    def put(self, key: str, value: Any):
        """Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        size = self._estimate_size(value)
        
        # Evict items if needed
        while self._total_size + size > self.max_size:
            self._evict()
            
        self._cache[key] = value
        self._sizes[key] = size
        self._timestamps[key] = time.time()
        
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._sizes.clear()
        self._timestamps.clear()
        if self.track_stats:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            "size": self._total_size,
            "items": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / (self._hits + self._misses) if self._hits + self._misses > 0 else 0
        }
        
    def evict(self, key: str):
        """Evict a specific key from the cache.
        
        Args:
            key: Key to evict
        """
        if key in self._cache:
            del self._cache[key]
            del self._sizes[key]
            del self._timestamps[key]
            if self.track_stats:
                self._evictions += 1
                
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes.
        
        Args:
            value: Value to estimate size of
            
        Returns:
            Estimated size in bytes
        """
        if hasattr(value, "nbytes"):
            return value.nbytes
        return len(str(value).encode())
        
    @property
    def _total_size(self) -> int:
        """Get the total size of the cache.
        
        Returns:
            Total size in bytes
        """
        return sum(self._sizes.values())
        
    def _evict(self):
        """Evict an item based on the eviction policy."""
        if not self._cache:
            return
            
        if self.eviction_policy == "lru":
            # Evict least recently used item
            key = min(self._timestamps.items(), key=lambda x: x[1])[0]
        else:
            # Evict first in first out item
            key = min(self._timestamps.items(), key=lambda x: x[1])[0]
            
        self.evict(key)

def get_available_memory() -> float:
    """Get the available memory in bytes.
    
    Returns:
        Available memory in bytes
    """
    return psutil.virtual_memory().available
