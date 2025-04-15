# JAX DataLoader Troubleshooting Guide

This guide provides detailed solutions for common issues encountered while using the JAX DataLoader.

## Table of Contents
1. [Memory Management](#memory-management)
2. [Performance Optimization](#performance-optimization)
3. [GPU/CUDA Issues](#gpu-cuda-issues)
4. [Data Loading Problems](#data-loading-problems)
5. [Distributed Training](#distributed-training)
6. [Data Augmentation](#data-augmentation)
7. [Progress Tracking](#progress-tracking)
8. [Caching System](#caching-system)
9. [Error Handling](#error-handling)
10. [Debugging Tools](#debugging-tools)

## Memory Management

### Out of Memory (OOM) Errors

**Symptoms:**
- Memory allocation failures
- Process killed by OOM killer
- CUDA out of memory errors

**Solutions:**

1. **Enable Memory Mapping**
   ```python
   from jax_dataloader import JAXDataLoader
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       use_mmap=True,  # Enable memory mapping
       mmap_mode='r'  # Read-only memory mapping
   )
   ```

2. **Reduce Memory Footprint**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=16,  # Reduce batch size
       num_workers=1,  # Reduce worker count
       prefetch_size=1,  # Reduce prefetch size
       use_pinned_memory=False  # Disable pinned memory
   )
   ```

3. **Monitor Memory Usage**
   ```python
   from jax_dataloader.memory import MemoryTracker
   
   with MemoryTracker() as tracker:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           debug=True
       )
       # Your code here
   tracker.print_report()
   ```

### Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- Memory not being released
- System slowdown

**Solutions:**

1. **Enable Memory Tracking**
   ```python
   from jax_dataloader.memory import MemoryTracker
   
   tracker = MemoryTracker(
       check_interval=100,  # Check every 100 batches
       threshold_mb=1024  # Alert if memory exceeds 1GB
   )
   
   with tracker:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           debug=True
       )
   ```

2. **Force Garbage Collection**
   ```python
   import gc
   from jax_dataloader.memory import cleanup_memory
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       cleanup_interval=1000  # Cleanup every 1000 batches
   )
   
   # Manual cleanup
   cleanup_memory()
   gc.collect()
   ```

## Performance Optimization

### Slow Data Loading

**Symptoms:**
- High latency between batches
- CPU bottleneck
- Disk I/O saturation

**Solutions:**

1. **Optimize Batch Processing**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=64,  # Increase batch size
       num_workers=4,  # Increase worker count
       prefetch_size=2,  # Increase prefetch size
       use_mmap=True,
       use_pinned_memory=True
   )
   ```

2. **Enable Caching**
   ```python
   from jax_dataloader.cache import LRUCache
   
   cache = LRUCache(
       max_size=1000,  # Cache 1000 batches
       prefetch=True  # Enable prefetching
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       cache=cache
   )
   ```

3. **Profile Performance**
   ```python
   from jax_dataloader.profile import PerformanceProfiler
   
   with PerformanceProfiler() as profiler:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32
       )
       # Your code here
   profiler.print_report()
   ```

### High CPU Usage

**Symptoms:**
- CPU saturation
- System slowdown
- High power consumption

**Solutions:**

1. **Optimize Worker Configuration**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       num_workers=2,  # Reduce worker count
       worker_type='thread',  # Use threads instead of processes
       use_mmap=True  # Reduce CPU overhead
   )
   ```

2. **Enable CPU Affinity**
   ```python
   from jax_dataloader.utils import set_cpu_affinity
   
   set_cpu_affinity([0, 1])  # Use specific CPU cores
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       num_workers=2
   )
   ```

## GPU/CUDA Issues

### GPU Memory Errors

**Symptoms:**
- CUDA out of memory errors
- GPU memory allocation failures
- Performance degradation

**Solutions:**

1. **Limit GPU Memory Usage**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       gpu_memory_fraction=0.8,  # Limit to 80% of GPU memory
       use_mmap=True,
       use_pinned_memory=True
   )
   ```

2. **Enable Memory Optimization**
   ```python
   from jax_dataloader.memory import GPUMemoryOptimizer
   
   optimizer = GPUMemoryOptimizer(
       max_memory_gb=8,  # Limit to 8GB
       cleanup_threshold=0.9  # Cleanup at 90% usage
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       memory_optimizer=optimizer
   )
   ```

### CUDA Device Errors

**Symptoms:**
- Device not found errors
- CUDA initialization failures
- Device synchronization issues

**Solutions:**

1. **Check Device Availability**
   ```python
   from jax_dataloader.utils import get_available_devices
   
   devices = get_available_devices()
   if not devices:
       print("No GPU devices found, falling back to CPU")
       import jax
       jax.config.update('jax_platform_name', 'cpu')
   else:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           device=devices[0]
       )
   ```

2. **Handle Device Errors**
   ```python
   from jax_dataloader.exceptions import DeviceError
   
   try:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           device='cuda:0'
       )
   except DeviceError as e:
       print(f"Device error: {e}")
       # Fallback to CPU
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           device='cpu'
       )
   ```

## Data Loading Problems

### Batch Shape Mismatch

**Symptoms:**
- Shape mismatch errors
- Inconsistent batch sizes
- Data type errors

**Solutions:**

1. **Validate Shapes**
   ```python
   from jax_dataloader.utils import validate_shapes
   
   validate_shapes(
       dataset,
       batch_size=32,
       expected_shape=(32, 224, 224, 3)
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32
   )
   ```

2. **Handle Dynamic Shapes**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       dynamic_shapes=True,  # Enable dynamic shape handling
       padding_value=0  # Pad with zeros if needed
   )
   ```

### Data Type Errors

**Symptoms:**
- Type conversion errors
- Incompatible data types
- Precision issues

**Solutions:**

1. **Specify Data Types**
   ```python
   import jax.numpy as jnp
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       dtype=jnp.float32,  # Specify data type
       convert_types=True  # Enable automatic conversion
   )
   ```

2. **Handle Mixed Types**
   ```python
   from jax_dataloader.utils import TypeConverter
   
   converter = TypeConverter(
       input_types={'image': jnp.float32, 'label': jnp.int32},
       output_types={'image': jnp.float16, 'label': jnp.int32}
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       type_converter=converter
   )
   ```

## Distributed Training

### Synchronization Errors

**Symptoms:**
- Deadlocks
- Timeout errors
- Inconsistent states

**Solutions:**

1. **Configure Synchronization**
   ```python
   from jax_dataloader.distributed import DistributedConfig
   
   config = DistributedConfig(
       num_nodes=2,
       node_rank=0,
       num_workers=4,
       sync_every_batch=True,
       barrier_timeout=30
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       distributed_config=config
   )
   ```

2. **Handle Timeouts**
   ```python
   from jax_dataloader.distributed import TimeoutHandler
   
   handler = TimeoutHandler(
       timeout=30,
       retry_attempts=3,
       backoff_factor=2
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       timeout_handler=handler
   )
   ```

### Load Balancing Issues

**Symptoms:**
- Uneven workload distribution
- Worker idle time
- Performance bottlenecks

**Solutions:**

1. **Enable Dynamic Balancing**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       dynamic_balancing=True,
       balance_interval=100,
       load_threshold=0.8
   )
   ```

2. **Monitor Worker Load**
   ```python
   from jax_dataloader.distributed import LoadMonitor
   
   monitor = LoadMonitor(
       check_interval=10,
       imbalance_threshold=0.2
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       load_monitor=monitor
   )
   ```

## Data Augmentation

### Augmentation Performance

**Symptoms:**
- Slow augmentation
- High CPU usage
- Memory issues

**Solutions:**

1. **Optimize Augmentation**
   ```python
   from jax_dataloader.transform import JAXDataAugmentation
   
   augmenter = JAXDataAugmentation(
       augmentations=['random_flip', 'random_rotation'],
       jit=True,
       parallel=True,
       num_workers=2
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       augmenter=augmenter
   )
   ```

2. **Cache Augmented Data**
   ```python
   from jax_dataloader.cache import AugmentationCache
   
   cache = AugmentationCache(
       max_size=1000,
       cache_augmented=True
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       augmentation_cache=cache
   )
   ```

## Progress Tracking

### Progress Bar Issues

**Symptoms:**
- Progress bar not updating
- Inaccurate progress
- Performance impact

**Solutions:**

1. **Configure Progress Tracking**
   ```python
   from jax_dataloader.progress import ProgressTracker
   
   tracker = ProgressTracker(
       total_batches=1000,
       update_interval=1,
       show_eta=True,
       format='{percentage}% | {bar} | {eta}'
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       progress_tracker=tracker
   )
   ```

2. **Custom Progress Format**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       progress_format='{percentage}% | {bar} | {eta} | {speed}',
       show_progress=True,
       progress_interval=1
   )
   ```

## Caching System

### Cache Memory Usage

**Symptoms:**
- High memory usage
- Cache thrashing
- Performance degradation

**Solutions:**

1. **Configure Cache Limits**
   ```python
   from jax_dataloader.cache import CacheMonitor
   
   monitor = CacheMonitor(
       max_memory_gb=2,
       cleanup_threshold=0.8,
       eviction_policy='lru'
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       cache_monitor=monitor
   )
   ```

2. **Optimize Cache Strategy**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       cache_size=1000,
       cache_strategy='lru',
       cache_prefetch=True,
       prefetch_size=2
   )
   ```

## Error Handling

### Common Exceptions

**Symptoms:**
- Various runtime errors
- Unexpected behavior
- System crashes

**Solutions:**

1. **Handle Common Errors**
   ```python
   from jax_dataloader.exceptions import (
       DataLoaderError,
       MemoryError,
       DeviceError
   )
   
   try:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32
       )
   except MemoryError:
       print("Memory error occurred, reducing batch size")
       loader = JAXDataLoader(
           dataset=data,
           batch_size=16
       )
   except DeviceError:
       print("Device error occurred, falling back to CPU")
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32,
           device='cpu'
       )
   except DataLoaderError as e:
       print(f"DataLoader error: {e}")
   ```

## Debugging Tools

### Enable Debug Mode

**Solutions:**

1. **Basic Debugging**
   ```python
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       debug=True,
       log_level='DEBUG'
   )
   ```

2. **Advanced Debugging**
   ```python
   from jax_dataloader.debug import Debugger
   
   debugger = Debugger(
       enable_memory_tracking=True,
       enable_performance_tracking=True,
       enable_error_tracking=True,
       log_file='debug.log'
   )
   
   loader = JAXDataLoader(
       dataset=data,
       batch_size=32,
       debugger=debugger
   )
   ```

### Performance Profiling

**Solutions:**

1. **Profile Data Loading**
   ```python
   from jax_dataloader.profile import DataLoaderProfiler
   
   profiler = DataLoaderProfiler(
       track_memory=True,
       track_performance=True,
       track_errors=True
   )
   
   with profiler:
       loader = JAXDataLoader(
           dataset=data,
           batch_size=32
       )
       # Your code here
   
   profiler.print_report()
   ```

2. **Generate Performance Report**
   ```python
   from jax_dataloader.profile import generate_report
   
   report = generate_report(
       loader,
       metrics=['memory', 'performance', 'errors'],
       format='html'  # or 'json', 'text'
   )
   report.save('performance_report.html')
   ``` 