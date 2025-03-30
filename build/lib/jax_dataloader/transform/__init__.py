"""Data transformation module for JAX applications."""

from typing import Any, Callable, Dict, List, Optional, Union
import jax.numpy as jnp

class Transform:
    """Base class for data transformations."""
    
    def __init__(self):
        """Initialize the transform."""
        self._transforms: List[Callable] = []
        
    def add(self, transform: Callable):
        """Add a transform function.
        
        Args:
            transform: Transform function to add
        """
        self._transforms.append(transform)
        
    def __call__(self, data: Any) -> Any:
        """Apply all transforms to the data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        for transform in self._transforms:
            data = transform(data)
        return data
