"""Utility functions for memory management and monitoring."""

import os
import psutil
import numpy as np
from typing import Dict, Optional, Tuple

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dict containing memory usage statistics in bytes.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss,
        'vms': memory_info.vms,
        'percent': process.memory_percent()
    }

def format_size(size_bytes: float) -> str:
    """Format size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human readable size string (e.g. "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get GPU memory usage statistics if available.
    
    Returns:
        Dict containing GPU memory usage statistics in bytes, or None if no GPU is available.
    """
    try:
        import jax
        devices = jax.devices()
        if not devices:
            return None
            
        memory_info = {}
        for i, device in enumerate(devices):
            memory_info[f'gpu_{i}'] = {
                'total': device.memory_size,
                'free': device.memory_free,
                'used': device.memory_size - device.memory_free
            }
        return memory_info
    except ImportError:
        return None 