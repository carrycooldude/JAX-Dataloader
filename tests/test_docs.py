import pytest
import jax
import jax.numpy as jnp
from jax_dataloader import DataLoader, DataLoaderConfig
from jax_dataloader.data import CSVLoader, JSONLoader, ImageLoader
from jax_dataloader.memory import MemoryManager, Cache
from jax_dataloader.progress import ProgressTracker
from jax_dataloader.transform import Transform
from jax_dataloader.exceptions import DataLoaderError, ConfigurationError, MemoryError
from jax_dataloader.utils import (
    get_available_memory,
    calculate_batch_size,
    get_device_count,
    format_size
)

def test_basic_dataloader():
    """Test basic DataLoader functionality."""
    data = jnp.arange(1000)
    labels = jnp.arange(1000)
    
    config = DataLoaderConfig(
        batch_size=32,
        shuffle=True
    )
    
    dataloader = DataLoader(
        data=data,
        labels=labels,
        config=config
    )
    
    for batch_data, batch_labels in dataloader:
        assert batch_data.shape[0] == 32
        assert batch_labels.shape[0] == 32
        break

def test_memory_management():
    """Test memory management features."""
    data = jnp.arange(1000000)
    
    config = DataLoaderConfig(
        batch_size=32,
        memory_fraction=0.8,
        auto_batch_size=True
    )
    
    dataloader = DataLoader(
        data=data,
        config=config
    )
    
    dataloader.optimize_memory()
    assert dataloader.memory_manager.get_usage() < 0.8

def test_caching():
    """Test caching functionality."""
    cache = Cache(
        max_size=1000,
        eviction_policy="lru"
    )
    
    # Test cache operations
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.hits == 1
    
    cache.put("key2", "value2")
    assert cache.get("key2") == "value2"
    assert cache.hits == 2

def test_progress_tracking():
    """Test progress tracking."""
    tracker = ProgressTracker(
        total=1000,
        update_interval=0.1
    )
    
    for i in range(100):
        tracker.update(1)
    
    assert tracker.get_progress() == 0.1
    assert tracker.get_eta() > 0

def test_data_augmentation():
    """Test data augmentation."""
    def augment_fn(batch, key):
        key1, key2 = random.split(key)
        noise = random.normal(key1, batch.shape) * 0.1
        angle = random.uniform(key2, minval=-0.1, maxval=0.1)
        return jnp.rot90(batch + noise, k=int(angle * 10))
    
    transform = Transform(
        fn=augment_fn,
        key=random.PRNGKey(0)
    )
    
    data = jnp.ones((10, 10))
    augmented = transform.apply(data)
    assert augmented.shape == data.shape

def test_error_handling():
    """Test error handling."""
    # Test DataLoaderError
    with pytest.raises(DataLoaderError):
        DataLoader(data=None)
    
    # Test ConfigurationError
    with pytest.raises(ConfigurationError):
        DataLoaderConfig(batch_size=-1)
    
    # Test MemoryError
    with pytest.raises(MemoryError):
        dataloader = DataLoader(data=jnp.ones((1000000, 1000000)))
        dataloader.optimize_memory()

def test_utility_functions():
    """Test utility functions."""
    # Test memory utilities
    memory = get_available_memory()
    assert memory > 0
    
    # Test batch size calculation
    batch_size = calculate_batch_size(
        total_size=10000,
        memory_fraction=0.8
    )
    assert batch_size > 0
    
    # Test device count
    num_devices = get_device_count()
    assert num_devices > 0
    
    # Test size formatting
    size = format_size(1024**3)
    assert size == "1.00 GB"

def test_data_loaders():
    """Test different data loaders."""
    # Test CSV loader
    csv_loader = CSVLoader(
        "test_data.csv",
        target_column="label",
        feature_columns=["feature1", "feature2"]
    )
    assert csv_loader is not None
    
    # Test JSON loader
    json_loader = JSONLoader(
        "test_data.json",
        data_key="features",
        label_key="labels"
    )
    assert json_loader is not None
    
    # Test Image loader
    image_loader = ImageLoader(
        "test_images",
        image_size=(224, 224),
        normalize=True
    )
    assert image_loader is not None

def test_multi_gpu():
    """Test multi-GPU functionality."""
    devices = jax.devices()
    
    config = DataLoaderConfig(
        batch_size=32,
        num_devices=len(devices),
        device_map="auto"
    )
    
    data = jnp.arange(10000)
    dataloader = DataLoader(
        data=data,
        config=config
    )
    
    for batch in dataloader:
        data, device_id = batch
        assert device_id < len(devices)
        break 