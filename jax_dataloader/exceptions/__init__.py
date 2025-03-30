"""Exception classes for JAX applications."""

class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass

class ConfigurationError(DataLoaderError):
    """Exception raised for configuration errors."""
    pass

class MemoryError(DataLoaderError):
    """Exception raised for memory-related errors."""
    pass
