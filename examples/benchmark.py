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

def benchmark_dataloaders(data_size=100000, feature_size=512, batch_size=256, num_epochs=5, device='cpu'):
    """Benchmark different data loading implementations"""
    try:
        pin_cpu_cores()
        
        # Generate synthetic data with reduced precision
        print("Generating dataset...")
        data = np.random.randn(data_size, feature_size).astype(np.float32)
        
        # Warm up GPU and CPU caches
        print("Warming up caches...")
        warmup_data = np.random.randn(1000, feature_size).astype(np.float32)
        if device == 'gpu':
            try:
                # Warm up JAX GPU
                _ = jax.device_put(warmup_data, device=jax.devices('gpu')[0])
                # Warm up PyTorch CUDA
                _ = torch.from_numpy(warmup_data).cuda()
                # Warm up TensorFlow GPU
                _ = tf.convert_to_tensor(warmup_data)
            except Exception as e:
                print(f"GPU warmup failed: {e}")
                device = 'cpu'
        
        results = {}
        
        # Benchmark JAX DataLoader
        print(f"\nBenchmarking JAX DataLoader on {device.upper()}...")
        try:
            jax_loader = JAXDataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,  # Single worker
                prefetch_size=1,  # Minimal prefetch
                use_mmap=True,
                use_pinned_memory=True,
                device=device
            )
            
            # Warmup
            for batch in jax_loader:
                if device == 'gpu':
                    _ = jax.device_put(batch, device=jax.devices('gpu')[0])
                break
                
            start_time = time.perf_counter()
            for epoch in tqdm(range(num_epochs), desc="JAX Epochs"):
                for batch in tqdm(jax_loader, desc="JAX Batches", leave=False):
                    # Simulate computation
                    if device == 'gpu':
                        batch = jax.device_put(batch, device=jax.devices('gpu')[0])
                    result = jax.jit(lambda x: jnp.mean(jnp.square(x)))(batch)
                    result.block_until_ready()
            end_time = time.perf_counter()
            results['JAX DataLoader'] = (end_time - start_time) / num_epochs
        except Exception as e:
            print(f"JAX benchmark failed: {e}")
            results['JAX DataLoader'] = float('inf')
        
        # Benchmark PyTorch DataLoader
        print(f"Benchmarking PyTorch DataLoader on {device.upper()}...")
        try:
            torch_data = torch.from_numpy(data)
            torch_dataset = torch.utils.data.TensorDataset(torch_data)
            torch_loader = torch.utils.data.DataLoader(
                torch_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,  # Single worker
                pin_memory=(device == 'gpu'),
                prefetch_factor=1  # Minimal prefetch
            )
            
            # Warmup
            for batch, in torch_loader:
                if device == 'gpu':
                    _ = batch.cuda(non_blocking=True)
                break
                
            start_time = time.perf_counter()
            for epoch in tqdm(range(num_epochs), desc="PyTorch Epochs"):
                for batch, in tqdm(torch_loader, desc="PyTorch Batches", leave=False):
                    if device == 'gpu':
                        batch = batch.cuda(non_blocking=True)
                    # Simulate computation
                    result = torch.mean(torch.square(batch))
                    if device == 'gpu':
                        torch.cuda.synchronize()
            end_time = time.perf_counter()
            results['PyTorch DataLoader'] = (end_time - start_time) / num_epochs
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            results['PyTorch DataLoader'] = float('inf')
        
        # Benchmark TensorFlow DataLoader
        print(f"Benchmarking TensorFlow DataLoader on {device.upper()}...")
        try:
            tf_data = tf.data.Dataset.from_tensor_slices(data)
            tf_loader = tf_data.shuffle(1000).batch(batch_size).prefetch(1)  # Minimal prefetch
            
            # Warmup
            for batch in tf_loader.take(1):
                if device == 'gpu':
                    _ = tf.identity(batch)
                break
                
            start_time = time.perf_counter()
            for epoch in tqdm(range(num_epochs), desc="TensorFlow Epochs"):
                for batch in tqdm(tf_loader, desc="TensorFlow Batches", leave=False):
                    # Simulate computation
                    result = tf.reduce_mean(tf.square(batch))
                    if device == 'gpu':
                        tf.experimental.async_scope.async_scope()
            end_time = time.perf_counter()
            results['TensorFlow DataLoader'] = (end_time - start_time) / num_epochs
        except Exception as e:
            print(f"TensorFlow benchmark failed: {e}")
            results['TensorFlow DataLoader'] = float('inf')
        
        # Benchmark NumPy (baseline)
        print(f"Benchmarking NumPy (baseline) on {device.upper()}...")
        try:
            indices = np.arange(len(data))
            
            start_time = time.perf_counter()
            for epoch in tqdm(range(num_epochs), desc="NumPy Epochs"):
                np.random.shuffle(indices)
                for i in tqdm(range(0, len(indices), batch_size), desc="NumPy Batches", leave=False):
                    batch_idx = indices[i:i + batch_size]
                    if len(batch_idx) < batch_size:
                        continue
                    batch = data[batch_idx]
                    # Simulate computation
                    result = np.mean(np.square(batch))
            end_time = time.perf_counter()
            results['NumPy'] = (end_time - start_time) / num_epochs
        except Exception as e:
            print(f"NumPy benchmark failed: {e}")
            results['NumPy'] = float('inf')
        
        # Print results
        print(f"\nResults on {device.upper()} (seconds per epoch):")
        for name, time_taken in results.items():
            if time_taken != float('inf'):
                print(f"{name:20s}: {time_taken:.4f}s")
            else:
                print(f"{name:20s}: Failed")
        
        # Calculate speedups
        baseline = results['NumPy']
        if baseline != float('inf'):
            print(f"\nSpeedup over NumPy baseline on {device.upper()}:")
            for name, time_taken in results.items():
                if name != 'NumPy' and time_taken != float('inf'):
                    speedup = baseline / time_taken
                    print(f"{name:20s}: {speedup:.2f}x faster")
        
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {}

def plot_benchmark_results(results: Dict[str, float], device: str):
    """Plot benchmark results with detailed visualizations"""
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Bar chart for absolute times
    ax1 = fig.add_subplot(gs[0, 0])
    frameworks = list(results.keys())
    times = list(results.values())
    
    bars = ax1.bar(frameworks, times, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    ax1.set_title(f'Time per Epoch ({device.upper()})', fontsize=14, pad=20)
    ax1.set_ylabel('Seconds', fontsize=12)
    ax1.set_xlabel('Framework', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Speedup chart
    ax2 = fig.add_subplot(gs[0, 1])
    baseline = results['NumPy']
    speedups = {k: baseline/v for k, v in results.items() if k != 'NumPy'}
    
    bars = ax2.bar(speedups.keys(), speedups.values(), color=['#4e79a7', '#f28e2b', '#e15759'])
    ax2.set_title(f'Speedup over NumPy ({device.upper()})', fontsize=14, pad=20)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_xlabel('Framework', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Memory usage chart
    ax3 = fig.add_subplot(gs[1, 0])
    memory_usage = [get_memory_usage() for _ in range(len(frameworks))]
    
    bars = ax3.bar(frameworks, memory_usage, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    ax3.set_title(f'Memory Usage ({device.upper()})', fontsize=14, pad=20)
    ax3.set_ylabel('Memory (MB)', fontsize=12)
    ax3.set_xlabel('Framework', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Throughput chart
    ax4 = fig.add_subplot(gs[1, 1])
    throughputs = {k: 1/v for k, v in results.items()}
    
    bars = ax4.bar(throughputs.keys(), throughputs.values(), color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    ax4.set_title(f'Throughput ({device.upper()})', fontsize=14, pad=20)
    ax4.set_ylabel('Batches per Second', fontsize=12)
    ax4.set_xlabel('Framework', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 5. Performance comparison table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create table data
    table_data = []
    for framework in frameworks:
        time_taken = results[framework]
        speedup = baseline/time_taken if framework != 'NumPy' else 1.0
        memory = get_memory_usage()
        throughput = 1/time_taken
        
        table_data.append([
            framework,
            f'{time_taken:.4f}s',
            f'{speedup:.2f}x',
            f'{memory:.1f} MB',
            f'{throughput:.2f}'
        ])
    
    # Create table
    table = ax5.table(
        cellText=table_data,
        colLabels=['Framework', 'Time/Epoch', 'Speedup', 'Memory', 'Throughput'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    ax5.set_title('Performance Comparison Summary', fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'benchmark_results_{device.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    df = pd.DataFrame(table_data, columns=['Framework', 'Time/Epoch', 'Speedup', 'Memory', 'Throughput'])
    df.to_csv(f'benchmark_results_{device.lower()}.csv', index=False)

def main():
    """Main function to run benchmarks"""
    parser = argparse.ArgumentParser(description='Benchmark JAX DataLoader')
    parser.add_argument('--data-size', type=int, default=100000,
                      help='Size of the dataset')
    parser.add_argument('--feature-size', type=int, default=512,
                      help='Number of features')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=5,
                      help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'gpu'],
                      help='Device to run benchmarks on')
    
    args = parser.parse_args()
    
    if jax.default_backend() == 'gpu':
        print("GPU detected, running GPU benchmark...")
        results = benchmark_dataloaders(
            data_size=args.data_size,
            feature_size=args.feature_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device='gpu'
        )
        plot_benchmark_results(results, 'gpu')
    else:
        print("No GPU detected, running CPU benchmark...")
        results = benchmark_dataloaders(
            data_size=args.data_size,
            feature_size=args.feature_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device='cpu'
        )
        plot_benchmark_results(results, 'cpu')

if __name__ == "__main__":
    main()