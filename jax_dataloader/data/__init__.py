"""Data loading module for JAX applications."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np

class BaseLoader(ABC):
    """Base class for all data loaders."""
    
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the data loader.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._rng = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        self._metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches."""
        pass
        
    @abstractmethod
    def __iter__(self):
        """Return an iterator over the data."""
        pass
        
    def __next__(self):
        """Get the next batch of data."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data loader.
        
        Returns:
            Dictionary containing metadata about the data loader
        """
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            **self._metadata
        }
        
    def load(self, data_path: str) -> Any:
        """Load data from the specified path.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded data
        """
        raise NotImplementedError("Subclasses must implement load()")
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess the loaded data.
        
        Args:
            data: Data to preprocess
            
        Returns:
            Preprocessed data
        """
        return data

class CSVLoader(BaseLoader):
    """Loader for CSV data."""
    
    def __init__(
        self,
        data_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        dtype: Any = jnp.float32,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the CSV loader.
        
        Args:
            data_path: Path to the CSV file
            target_column: Name of the target column
            feature_columns: List of feature column names
            chunk_size: Size of chunks to load at once
            dtype: Data type for the loaded data
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__(batch_size, shuffle, seed, num_workers, prefetch_factor)
        self.data_path = data_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.chunk_size = chunk_size
        self.dtype = dtype
        self._data = None
        self._labels = None
        
    def load(self, data_path: str) -> Any:
        """Load CSV data from the specified path.
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            Loaded CSV data
        """
        df = pd.read_csv(data_path)
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in CSV")
            
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
            
        data = df[self.feature_columns].values.astype(self.dtype)
        labels = df[self.target_column].values
        
        self._data = data
        self._labels = labels
        return data, labels
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess the loaded CSV data.
        
        Args:
            data: CSV data to preprocess
            
        Returns:
            Preprocessed data
        """
        return data
        
    def get_chunk(self, start_idx: int, chunk_size: Optional[int] = None) -> Tuple[Any, Any]:
        """Get a chunk of data.
        
        Args:
            start_idx: Starting index
            chunk_size: Size of the chunk to load
            
        Returns:
            Tuple of (data chunk, label chunk)
        """
        if chunk_size is None:
            chunk_size = self.chunk_size or len(self._data)
            
        end_idx = min(start_idx + chunk_size, len(self._data))
        return self._data[start_idx:end_idx], self._labels[start_idx:end_idx]
        
    def __len__(self) -> int:
        """Return the number of batches."""
        if self._data is None:
            self.load(self.data_path)
        return (len(self._data) + self.batch_size - 1) // self.batch_size
        
    def __iter__(self):
        """Return an iterator over the data."""
        if self._data is None:
            self.load(self.data_path)
            
        indices = np.arange(len(self._data))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self._data[batch_indices], self._labels[batch_indices]

class JSONLoader(BaseLoader):
    """Loader for JSON data."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the JSON loader.
        
        Args:
            data_path: Path to the JSON file
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__(batch_size, shuffle, seed, num_workers, prefetch_factor)
        self.data_path = data_path
        # TODO: Implement JSON loading logic
        
    def load(self, data_path: str) -> Any:
        """Load JSON data from the specified path.
        
        Args:
            data_path: Path to the JSON file
            
        Returns:
            Loaded JSON data
        """
        import json
        with open(data_path, 'r') as f:
            return json.load(f)
            
    def preprocess(self, data: Any) -> Any:
        """Preprocess the loaded JSON data.
        
        Args:
            data: JSON data to preprocess
            
        Returns:
            Preprocessed data
        """
        return data

class ImageLoader(BaseLoader):
    """Loader for image data."""
    
    def __init__(
        self,
        data_path: str,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        augment_options: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the image loader.
        
        Args:
            data_path: Path to the image directory
            image_size: Target size for images
            normalize: Whether to normalize pixel values
            augment: Whether to apply data augmentation
            augment_options: Options for data augmentation
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__(batch_size, shuffle, seed, num_workers, prefetch_factor)
        self.data_path = data_path
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.augment_options = augment_options or {}
        self._image_files = []
        self._labels = []
        
    def load(self, data_path: str) -> Any:
        """Load image data from the specified path.
        
        Args:
            data_path: Path to the image directory
            
        Returns:
            Loaded image data
        """
        import os
        from PIL import Image
        
        self._image_files = []
        self._labels = []
        
        for filename in os.listdir(data_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self._image_files.append(os.path.join(data_path, filename))
                # Assuming labels are part of filename (e.g., "0_image1.jpg")
                label = int(filename.split('_')[0])
                self._labels.append(label)
                
        return self._image_files, self._labels
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess the loaded image data.
        
        Args:
            data: Image data to preprocess
            
        Returns:
            Preprocessed data
        """
        from PIL import Image
        import numpy as np
        
        if isinstance(data, str):
            img = Image.open(data)
            img = img.resize(self.image_size)
            img_array = np.array(img)
            
            if self.normalize:
                img_array = img_array.astype(np.float32) / 255.0
                
            return img_array
        return data
        
    def augment(self, data: Any) -> Any:
        """Apply data augmentation to the image.
        
        Args:
            data: Image data to augment
            
        Returns:
            Augmented image data
        """
        if not self.augment:
            return data
            
        import jax.random as random
        key = random.PRNGKey(0)
        
        # Apply random rotation
        if "rotation" in self.augment_options:
            angle = random.uniform(
                key,
                minval=self.augment_options["rotation"][0],
                maxval=self.augment_options["rotation"][1]
            )
            data = jnp.rot90(data, k=int(angle))
            
        # Apply random flip
        if self.augment_options.get("flip", False):
            key, subkey = random.split(key)
            if random.uniform(subkey) > 0.5:
                data = jnp.fliplr(data)
                
        # Apply brightness adjustment
        if "brightness" in self.augment_options:
            key, subkey = random.split(key)
            factor = random.uniform(
                subkey,
                minval=self.augment_options["brightness"][0],
                maxval=self.augment_options["brightness"][1]
            )
            data = data * factor
            
        return data
        
    def __len__(self) -> int:
        """Return the number of batches."""
        if not self._image_files:
            self.load(self.data_path)
        return (len(self._image_files) + self.batch_size - 1) // self.batch_size
        
    def __iter__(self):
        """Return an iterator over the data."""
        if not self._image_files:
            self.load(self.data_path)
            
        indices = np.arange(len(self._image_files))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img = self.preprocess(self._image_files[idx])
                if self.augment:
                    img = self.augment(img)
                batch_images.append(img)
                batch_labels.append(self._labels[idx])
                
            yield jnp.array(batch_images), jnp.array(batch_labels)

def get_device_count() -> int:
    """Get the number of available devices.
    
    Returns:
        Number of available devices
    """
    # TODO: Implement device counting logic
    return 1
