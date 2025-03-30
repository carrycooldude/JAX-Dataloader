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
        
    def apply(self, data: Any) -> Any:
        """Apply all transforms to the data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        return self(data)
        
    def compose(self, other: 'Transform') -> 'Transform':
        """Compose this transform with another transform.
        
        Args:
            other: Transform to compose with
            
        Returns:
            New composed transform
        """
        result = Transform()
        result._transforms = self._transforms + other._transforms
        return result
        
    def chain(self, *transforms: Callable) -> 'Transform':
        """Chain multiple transforms together.
        
        Args:
            *transforms: Transforms to chain
            
        Returns:
            New transform with all transforms chained
        """
        result = Transform()
        result._transforms = self._transforms + list(transforms)
        return result
