"""Utility functions for JAX applications."""

import os
from typing import Union
import psutil
from jax.lib import xla_bridge

def format_size(size: Union[int, float]) -> str:
    """Format a size in bytes to a human-readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def calculate_batch_size(
    total_size: int,
    max_memory: float,
    sample_size: int,
) -> int:
    """Calculate the optimal batch size based on available memory.
    
    Args:
        total_size: Total size of the dataset
        max_memory: Maximum available memory in bytes
        sample_size: Size of a single sample in bytes
        
    Returns:
        Optimal batch size
    """
    # Leave some memory for other operations
    available_memory = max_memory * 0.8
    
    # Calculate maximum batch size based on memory
    max_batch_size = int(available_memory / sample_size)
    
    # Ensure batch size is not too large
    return min(max_batch_size, total_size)

def get_available_memory() -> float:
    """Get the available memory in bytes.
    
    Returns:
        Available memory in bytes
    """
    return psutil.virtual_memory().available

def get_device_count() -> int:
    """Get the number of available devices.
    
    Returns:
        Number of available devices
    """
    return xla_bridge.device_count()
