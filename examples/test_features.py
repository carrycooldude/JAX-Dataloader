import jax
import jax.numpy as jnp
import numpy as np
from jax_dataloader import JAXDataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import os
import psutil

def test_memory_management():
    print("\n=== Testing Memory Management ===")
    # Create a large dataset
    data = np.random.randn(10000, 512).astype(np.float32)
    
    # Initialize with memory management
    loader = JAXDataLoader(
        dataset=data,
        batch_size=256,
        num_workers=4,
        use_mmap=True,
        use_pinned_memory=True
    )
    
    # Test memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Process some batches
    for i, batch in enumerate(loader):
        if i >= 5:  # Process 5 batches
            break
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Batch {i+1} memory usage: {current_memory:.2f} MB")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory overhead: {final_memory - initial_memory:.2f} MB")

def test_distributed_training():
    print("\n=== Testing Distributed Training ===")
    # Load IMDB dataset
    dataset = load_dataset("imdb", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize distributed loader
    loader = JAXDataLoader(
        dataset=dataset,
        batch_size=32,
        tokenizer=tokenizer,
        distributed=True,
        num_workers=4,
        world_size=2,
        rank=0
    )
    
    # Test distributed processing
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:  # Process 5 batches
            break
        print(f"Processed batch {i+1} with shape: {batch['input_ids'].shape}")
    
    elapsed = time.time() - start_time
    print(f"Distributed processing time: {elapsed:.2f} seconds")

def test_data_augmentation():
    print("\n=== Testing Data Augmentation ===")
    # Create image-like data
    images = np.random.randn(100, 32, 32, 3).astype(np.float32)
    
    # Initialize with augmentations
    loader = JAXDataLoader(
        dataset=images,
        batch_size=16,
        augmentations=[
            "random_flip",
            "color_jitter",
            "gaussian_noise"
        ]
    )
    
    # Test augmentation
    for i, batch in enumerate(loader):
        if i >= 3:  # Process 3 batches
            break
        print(f"Batch {i+1} shape: {batch.shape}")
        print(f"Batch {i+1} stats - min: {batch.min():.3f}, max: {batch.max():.3f}, mean: {batch.mean():.3f}")

def test_progress_logging():
    print("\n=== Testing Progress Logging ===")
    # Create dataset
    data = np.random.randn(1000, 128).astype(np.float32)
    
    # Initialize with logging
    loader = JAXDataLoader(
        dataset=data,
        batch_size=64,
        log_file="test_training.log"
    )
    
    # Test progress tracking
    loader.start_epoch(0, 10)
    for i, batch in enumerate(loader):
        if i >= 5:  # Process 5 batches
            break
        time.sleep(0.1)  # Simulate processing time
    
    # Display summary
    loader.display_summary()
    loader.save_metrics("test_metrics.json")

def test_huggingface_integration():
    print("\n=== Testing HuggingFace Integration ===")
    # Load dataset and tokenizer
    dataset = load_dataset("glue", "sst2", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize loader
    loader = JAXDataLoader(
        dataset=dataset,
        batch_size=32,
        tokenizer=tokenizer,
        shuffle=True
    )
    
    # Test processing
    for i, batch in enumerate(loader):
        if i >= 3:  # Process 3 batches
            break
        print(f"Batch {i+1} keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")

if __name__ == "__main__":
    # Run all tests
    test_memory_management()
    test_distributed_training()
    test_data_augmentation()
    test_progress_logging()
    test_huggingface_integration() 