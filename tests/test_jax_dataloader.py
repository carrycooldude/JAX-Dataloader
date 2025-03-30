import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jax_dataloader import JAXDataLoader, DataLoaderConfig, load_custom_data
import tempfile
import os
import pandas as pd
import json
from PIL import Image
import shutil

@pytest.fixture
def sample_data():
    # Create sample data
    data = np.random.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
    labels = np.random.randint(0, 10, 100)  # 10 classes
    return data, labels

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_basic_functionality(sample_data):
    """Test basic dataloader functionality."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=32, shuffle=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    # Test iteration
    for batch_x, batch_y in dataloader:
        assert batch_x.shape[0] == 32
        assert batch_y.shape[0] == 32
        break

def test_memory_management(sample_data):
    """Test memory management features."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=32, pinned_memory=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    # Test memory allocation
    initial_memory = dataloader.memory_manager.allocated_memory
    batch_x, batch_y = next(iter(dataloader))
    assert dataloader.memory_manager.allocated_memory > initial_memory
    
    # Test memory deallocation
    del batch_x, batch_y
    dataloader._cleanup()
    assert dataloader.memory_manager.allocated_memory == 0

def test_caching(sample_data):
    """Test data caching functionality."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=32, cache_size=50)
    dataloader = JAXDataLoader(data, labels, config)
    
    # First pass - cache should be empty
    assert len(dataloader.cache) == 0
    
    # Load some batches
    for _ in range(2):
        next(iter(dataloader))
    
    # Cache should contain some items
    assert len(dataloader.cache) > 0
    assert len(dataloader.cache) <= config.cache_size

def test_augmentation(sample_data):
    """Test data augmentation."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=32, augmentation=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    # Get original and augmented batch
    original_batch = data[:32]
    augmented_batch, _ = next(iter(dataloader))
    
    # Check that some samples were augmented
    assert not np.array_equal(original_batch, augmented_batch)

def test_multi_gpu(sample_data):
    """Test multi-GPU support."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=64, multi_gpu=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    batch_x, batch_y = next(iter(dataloader))
    num_devices = jax.device_count()
    
    # Check batch distribution
    assert batch_x.shape[0] == num_devices
    assert batch_x.shape[1] == 64 // num_devices

def test_progress_tracking(sample_data):
    """Test progress tracking."""
    data, labels = sample_data
    config = DataLoaderConfig(batch_size=32, progress_tracking=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    # Load some batches
    for _ in range(2):
        next(iter(dataloader))
    
    stats = dataloader.get_stats()
    assert stats['progress']['batches_processed'] > 0
    assert stats['progress']['start_time'] is not None

def test_auto_batch_size(sample_data):
    """Test automatic batch size tuning."""
    data, labels = sample_data
    # Create a very large batch size that should be reduced
    config = DataLoaderConfig(batch_size=10000, auto_batch_size=True)
    dataloader = JAXDataLoader(data, labels, config)
    
    # Batch size should be reduced based on available memory
    assert dataloader.config.batch_size < 10000
    assert dataloader.config.batch_size > 0
    assert dataloader.config.batch_size % dataloader.num_devices == 0  # Should be divisible by number of devices

def test_csv_loading(temp_dir):
    """Test loading data from CSV."""
    # Create sample CSV
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'label': np.random.randint(0, 10, 100)
    })
    csv_path = os.path.join(temp_dir, 'test.csv')
    df.to_csv(csv_path, index=False)
    
    dataloader = load_custom_data(csv_path, file_type='csv', target_column='label')
    batch_x, batch_y = next(iter(dataloader))
    
    assert batch_x.shape[1] == 2  # 2 features
    assert batch_y.shape[0] == dataloader.config.batch_size

def test_json_loading(temp_dir):
    """Test loading data from JSON."""
    # Create sample JSON
    data = [
        {'features': np.random.randn(10).tolist(), 'label': int(np.random.randint(0, 10))}
        for _ in range(100)
    ]
    json_path = os.path.join(temp_dir, 'test.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    dataloader = load_custom_data(json_path, file_type='json')
    batch_x, batch_y = next(iter(dataloader))
    
    assert batch_x.shape[1] == 10  # 10 features
    assert batch_y.shape[0] == dataloader.config.batch_size

def test_image_loading(temp_dir):
    """Test loading image data."""
    # Create sample images
    img_dir = os.path.join(temp_dir, 'images')
    os.makedirs(img_dir)
    
    for i in range(100):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(os.path.join(img_dir, f'{i}_image.jpg'))
    
    dataloader = load_custom_data(img_dir, file_type='image')
    batch_x, batch_y = next(iter(dataloader))
    
    assert batch_x.shape[1:] == (64, 64, 3)  # Image dimensions
    assert batch_y.shape[0] == dataloader.config.batch_size

def test_error_handling(sample_data):
    """Test error handling."""
    _, labels = sample_data
    
    # Test with invalid data
    with pytest.raises(ValueError):
        JAXDataLoader(None, labels)
    
    # Test with mismatched data and labels
    with pytest.raises(ValueError):
        JAXDataLoader(np.random.randn(50, 3, 32, 32), labels) 