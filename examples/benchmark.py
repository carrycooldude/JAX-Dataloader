import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax_dataloader import JAXDataLoader
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import gc
import psutil
from jax import profiler
import seaborn as sns
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import argparse

# Enable GPU if available
if jax.default_backend() == 'gpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Enable first GPU
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Prevent OOM errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow warnings
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU if not available

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def create_dataset(size: int, dim: int = 32, dtype: str = 'float32') -> np.ndarray:
    """Create a synthetic dataset with specified type"""
    return np.random.rand(size, dim).astype(dtype)

def pin_cpu_cores():
    """Pin process to specific CPU cores for consistent benchmarking"""
    cpu_count = psutil.cpu_count(logical=False)
    os.sched_setaffinity(0, range(cpu_count))

def benchmark_dataloaders(data_size=100000, feature_size=1024, batch_size=32, num_epochs=5, device='cpu'):
    """Benchmark different data loading implementations"""
    pin_cpu_cores()
    
    # Generate synthetic data
    data = np.random.randn(data_size, feature_size).astype(np.float32)
    
    # Warm up GPU and CPU caches
    warmup_data = np.random.randn(1000, feature_size).astype(np.float32)
    if device == 'gpu':
        # Warm up JAX GPU
        _ = jax.device_put(warmup_data, device=jax.devices('gpu')[0])
        # Warm up PyTorch CUDA
        _ = torch.from_numpy(warmup_data).cuda()
        # Warm up TensorFlow GPU
        _ = tf.convert_to_tensor(warmup_data)
    
    results = {}
    
    # Benchmark JAX DataLoader with Python backend
    print(f"\nBenchmarking JAX DataLoader on {device.upper()}...")
    jax_loader = JAXDataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Warmup
    for batch in jax_loader:
        if device == 'gpu':
            _ = jax.device_put(batch, device=jax.devices('gpu')[0])
        break
        
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for batch in jax_loader:
            # Simulate computation
            if device == 'gpu':
                batch = jax.device_put(batch, device=jax.devices('gpu')[0])
            result = jax.jit(lambda x: jnp.mean(jnp.square(x)))(batch)
            result.block_until_ready()
    end_time = time.perf_counter()
    results['JAX DataLoader'] = (end_time - start_time) / num_epochs
    
    # Benchmark PyTorch DataLoader
    print(f"Benchmarking PyTorch DataLoader on {device.upper()}...")
    torch_data = torch.from_numpy(data)
    torch_dataset = torch.utils.data.TensorDataset(torch_data)
    torch_loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == 'gpu'),  # Enable pinned memory for GPU
        prefetch_factor=2
    )
    
    # Warmup
    for batch, in torch_loader:
        if device == 'gpu':
            _ = batch.cuda(non_blocking=True)
        break
        
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for batch, in torch_loader:
            if device == 'gpu':
                batch = batch.cuda(non_blocking=True)
            # Simulate computation
            result = torch.mean(torch.square(batch))
            if device == 'gpu':
                torch.cuda.synchronize()
    end_time = time.perf_counter()
    results['PyTorch DataLoader'] = (end_time - start_time) / num_epochs
    
    # Benchmark TensorFlow DataLoader
    print(f"Benchmarking TensorFlow DataLoader on {device.upper()}...")
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    tf_loader = tf_data.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Warmup
    for batch in tf_loader.take(1):
        if device == 'gpu':
            _ = tf.identity(batch)
        break
        
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for batch in tf_loader:
            # Simulate computation
            result = tf.reduce_mean(tf.square(batch))
            if device == 'gpu':
                tf.experimental.async_scope.async_scope()
    end_time = time.perf_counter()
    results['TensorFlow DataLoader'] = (end_time - start_time) / num_epochs
    
    # Benchmark NumPy (baseline)
    print(f"Benchmarking NumPy (baseline) on {device.upper()}...")
    indices = np.arange(len(data))
    
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < batch_size:
                continue
            batch = data[batch_idx]
            # Simulate computation
            result = np.mean(np.square(batch))
    end_time = time.perf_counter()
    results['NumPy'] = (end_time - start_time) / num_epochs
    
    # Print results
    print(f"\nResults on {device.upper()} (seconds per epoch):")
    for name, time_taken in results.items():
        print(f"{name:20s}: {time_taken:.4f}s")
    
    # Calculate speedups
    baseline = results['NumPy']
    print(f"\nSpeedup over NumPy baseline on {device.upper()}:")
    for name, time_taken in results.items():
        if name != 'NumPy':
            speedup = baseline / time_taken
            print(f"{name:20s}: {speedup:.2f}x faster")

def benchmark_loader(
    loader,
    num_epochs: int = 3,
    warmup_epochs: int = 1,
    device: str = 'cpu',
    profile: bool = False
) -> Tuple[float, float, List[float]]:
    """Benchmark a data loader with memory profiling"""
    memory_usage = []
    
    # Warmup
    for _ in range(warmup_epochs):
        for batch in loader:
            if device == 'gpu':
                if isinstance(batch, torch.Tensor):
                    batch = batch.cuda()
                elif isinstance(batch, np.ndarray):
                    batch = jax.device_put(batch, device=jax.devices('gpu')[0])
            _ = batch
        memory_usage.append(get_memory_usage())

    # Actual benchmark
    start_time = time.time()
    if profile and isinstance(loader, JAXDataLoader):
        profiler.start_trace("jax_trace")
    
    for _ in range(num_epochs):
        for batch in loader:
            if device == 'gpu':
                if isinstance(batch, torch.Tensor):
                    batch = batch.cuda()
                elif isinstance(batch, np.ndarray):
                    batch = jax.device_put(batch, device=jax.devices('gpu')[0])
            _ = batch
        memory_usage.append(get_memory_usage())
    
    if profile and isinstance(loader, JAXDataLoader):
        profiler.stop_trace()
    
    end_time = time.time()
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    return end_time - start_time, avg_memory, memory_usage

def run_benchmarks(
    dataset_sizes: List[int] = [10_000, 100_000, 1_000_000],
    batch_sizes: List[int] = [32, 64, 128, 256],
    data_types: List[str] = ['float32', 'float64'],
    device: str = 'cpu',
    num_workers_list: List[int] = [2, 4, 8, 16],
    profile: bool = False
) -> Dict[str, Dict[str, List[float]]]:
    """Run comprehensive benchmarks"""
    results = {
        'jax': {
            'times': [], 'throughputs': [], 'memory': [],
            'workers': [], 'batch_sizes': [], 'data_types': []
        },
        'tf': {
            'times': [], 'throughputs': [], 'memory': [],
            'workers': [], 'batch_sizes': [], 'data_types': []
        },
        'torch': {
            'times': [], 'throughputs': [], 'memory': [],
            'workers': [], 'batch_sizes': [], 'data_types': []
        }
    }

    for size in dataset_sizes:
        print(f"\nBenchmarking with dataset size: {size:,}")
        
        for dtype in data_types:
            print(f"\nData type: {dtype}")
            dataset = create_dataset(size, dtype=dtype)
            
            for batch_size in batch_sizes:
                print(f"\nBatch size: {batch_size}")
                
                for num_workers in num_workers_list:
                    print(f"Workers: {num_workers}")
                    
                    # JAX DataLoader
                    jax_loader = JAXDataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers
                    )
                    jax_time, jax_memory, _ = benchmark_loader(
                        jax_loader, device=device, profile=profile
                    )
                    jax_throughput = (size * 3) / jax_time
                    
                    results['jax']['times'].append(jax_time)
                    results['jax']['throughputs'].append(jax_throughput)
                    results['jax']['memory'].append(jax_memory)
                    results['jax']['workers'].append(num_workers)
                    results['jax']['batch_sizes'].append(batch_size)
                    results['jax']['data_types'].append(dtype)
                    
                    print(f"JAX: {jax_time:.2f}s ({jax_throughput:.2f} samples/sec, {jax_memory:.2f} MB)")
                    del jax_loader
                    gc.collect()

                    # TensorFlow
                    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
                    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    tf_time, tf_memory, _ = benchmark_loader(tf_dataset, device=device)
                    tf_throughput = (size * 3) / tf_time
                    
                    results['tf']['times'].append(tf_time)
                    results['tf']['throughputs'].append(tf_throughput)
                    results['tf']['memory'].append(tf_memory)
                    results['tf']['workers'].append(num_workers)
                    results['tf']['batch_sizes'].append(batch_size)
                    results['tf']['data_types'].append(dtype)
                    
                    print(f"TF: {tf_time:.2f}s ({tf_throughput:.2f} samples/sec, {tf_memory:.2f} MB)")
                    del tf_dataset
                    gc.collect()

                    # PyTorch
                    torch_dataset = TensorDataset(torch.from_numpy(dataset))
                    torch_loader = DataLoader(
                        torch_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True
                    )
                    torch_time, torch_memory, _ = benchmark_loader(torch_loader, device=device)
                    torch_throughput = (size * 3) / torch_time
                    
                    results['torch']['times'].append(torch_time)
                    results['torch']['throughputs'].append(torch_throughput)
                    results['torch']['memory'].append(torch_memory)
                    results['torch']['workers'].append(num_workers)
                    results['torch']['batch_sizes'].append(batch_size)
                    results['torch']['data_types'].append(dtype)
                    
                    print(f"PyTorch: {torch_time:.2f}s ({torch_throughput:.2f} samples/sec, {torch_memory:.2f} MB)")
                    del torch_loader
                    gc.collect()

    return results

def plot_results(results: Dict[str, Dict[str, List[float]]]):
    """Create comprehensive visualizations"""
    # Convert results to DataFrame for easier plotting
    dfs = []
    for framework in ['jax', 'tf', 'torch']:
        df = pd.DataFrame({
            'Framework': framework,
            'Time': results[framework]['times'],
            'Throughput': results[framework]['throughputs'],
            'Memory': results[framework]['memory'],
            'Workers': results[framework]['workers'],
            'Batch Size': results[framework]['batch_sizes'],
            'Data Type': results[framework]['data_types']
        })
        dfs.append(df)
    
    df = pd.concat(dfs)
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 15))
    
    # 1. Throughput vs Batch Size
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df, x='Batch Size', y='Throughput', hue='Framework', style='Data Type')
    plt.title('Throughput vs Batch Size')
    plt.yscale('log')
    
    # 2. Memory Usage vs Workers
    plt.subplot(2, 2, 2)
    sns.lineplot(data=df, x='Workers', y='Memory', hue='Framework', style='Data Type')
    plt.title('Memory Usage vs Number of Workers')
    
    # 3. Time vs Batch Size
    plt.subplot(2, 2, 3)
    sns.lineplot(data=df, x='Batch Size', y='Time', hue='Framework', style='Data Type')
    plt.title('Loading Time vs Batch Size')
    plt.yscale('log')
    
    # 4. Throughput vs Memory Usage
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='Memory', y='Throughput', hue='Framework', style='Data Type')
    plt.title('Throughput vs Memory Usage')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    df.to_csv('benchmark_results.csv', index=False)

def benchmark_framework(framework: str, data: np.ndarray, batch_size: int, device: str, num_epochs: int = 3) -> float:
    """Benchmark a single framework with consistent warmup and synchronization"""
    gc.collect()  # Clean up memory before benchmark
    
    if framework == 'jax':
        # Pre-compile JAX operations and move data to device
        data = jnp.asarray(data)
        if device == 'gpu':
            data = jax.device_put(data, jax.devices('gpu')[0])

        # Pre-compile computation
        @jax.jit
        def compute_mean(x):
            return jnp.mean(jnp.square(x))
        
        loader = JAXDataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Increased number of workers
            prefetch_size=4,  # Increased prefetch size
            use_mmap=True  # Enable memory mapping
        )
        
        # Warmup JIT and data loading
        for batch in loader:
            if device == 'gpu':
                batch = jax.device_put(batch, jax.devices('gpu')[0])
            _ = compute_mean(batch).block_until_ready()
            break
            
        # Benchmark with proper synchronization
        start_time = time.perf_counter()
        for _ in range(num_epochs):
            for batch in loader:
                if device == 'gpu':
                    batch = jax.device_put(batch, jax.devices('gpu')[0])
                result = compute_mean(batch)
                result.block_until_ready()  # Ensure computation is complete
        end_time = time.perf_counter()
        
    elif framework == 'torch':
        # Move data to device first
        torch_data = torch.from_numpy(data)
        if device == 'gpu':
            torch_data = torch_data.cuda()
            
        torch_dataset = torch.utils.data.TensorDataset(torch_data)
        loader = torch.utils.data.DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Increased number of workers
            pin_memory=(device == 'gpu'),
            prefetch_factor=4  # Increased prefetch factor
        )
        
        # Warmup
        for batch, in loader:
            if device == 'gpu':
                batch = batch.cuda(non_blocking=True)
            _ = torch.mean(torch.square(batch))
            break
            
        start_time = time.perf_counter()
        for _ in range(num_epochs):
            for batch, in loader:
                if device == 'gpu':
                    batch = batch.cuda(non_blocking=True)
                result = torch.mean(torch.square(batch))
                if device == 'gpu':
                    torch.cuda.synchronize()
        end_time = time.perf_counter()
        
    elif framework == 'tf':
        # Move data to device first
        if device == 'gpu':
            with tf.device('/GPU:0'):
                tf_data = tf.convert_to_tensor(data)
        else:
            tf_data = tf.convert_to_tensor(data)
            
        dataset = tf.data.Dataset.from_tensor_slices(tf_data)
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Warmup
        for batch in dataset.take(1):
            _ = tf.reduce_mean(tf.square(batch))
            
        start_time = time.perf_counter()
        for _ in range(num_epochs):
            for batch in dataset:
                result = tf.reduce_mean(tf.square(batch))
                if device == 'gpu':
                    tf.experimental.async_scope.async_scope()
        end_time = time.perf_counter()
        
    else:  # numpy
        # Ensure data is in numpy array
        data = np.asarray(data)
        indices = np.arange(len(data))
        
        start_time = time.perf_counter()
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                if len(batch_idx) < batch_size:
                    continue
                batch = data[batch_idx]
                result = np.mean(np.square(batch))
        end_time = time.perf_counter()
    
    return (end_time - start_time) / num_epochs

def plot_benchmark_results(results: Dict[str, float], device: str):
    """Plot benchmark results with visualizations"""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart for absolute times
    frameworks = list(results.keys())
    times = list(results.values())
    
    bars = ax1.bar(frameworks, times, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    ax1.set_title(f'Time per Epoch ({device.upper()})')
    ax1.set_ylabel('Seconds')
    ax1.set_xlabel('Framework')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom')
    
    # Speedup chart
    baseline = results['numpy']
    speedups = {k: baseline/v for k, v in results.items() if k != 'numpy'}
    
    bars = ax2.bar(speedups.keys(), speedups.values(), color=['#4e79a7', '#f28e2b', '#e15759'])
    ax2.set_title(f'Speedup over NumPy ({device.upper()})')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_xlabel('Framework')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'benchmark_results_{device.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_quick_benchmark(data_size=500000, feature_size=1024, batch_size=512, device='cpu'):
    """Run a quick benchmark with memory-efficient parameters"""
    print(f"\nRunning quick benchmark on {device.upper()}...")
    print(f"Dataset size: {data_size}, Feature size: {feature_size}, Batch size: {batch_size}")
    
    # Generate random data in chunks to avoid OOM
    chunk_size = 100000
    data = np.zeros((data_size, feature_size), dtype=np.float32)
    for i in range(0, data_size, chunk_size):
        end = min(i + chunk_size, data_size)
        data[i:end] = np.random.randn(end - i, feature_size).astype(np.float32)
    
    # Run benchmarks
    results = {}
    for framework in ['jax', 'torch', 'tf', 'numpy']:
        time_taken = benchmark_framework(framework, data, batch_size, device)
        results[framework] = time_taken
        
    # Print results
    print("\nResults (seconds per epoch):")
    for framework, time_taken in results.items():
        print(f"{framework:20}: {time_taken:.4f}s")
        
    # Calculate speedup over NumPy baseline
    numpy_time = results['numpy']
    print("\nSpeedup over NumPy baseline:")
    for framework, time_taken in results.items():
        if framework != 'numpy':
            speedup = numpy_time / time_taken
            print(f"{framework:20}: {speedup:.2f}x faster")
            
    # Plot results
    plot_benchmark_results(results, device)

def main():
    """Main function to run benchmarks"""
    if jax.default_backend() == 'gpu':
        print("GPU detected, running GPU benchmark...")
        run_quick_benchmark(data_size=500000, feature_size=1024, batch_size=512, device='gpu')
    else:
        print("No GPU detected, running CPU benchmark...")
        run_quick_benchmark(data_size=500000, feature_size=1024, batch_size=512, device='cpu')

if __name__ == "__main__":
    main()