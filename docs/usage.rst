Usage Guide
===========

This guide covers the basic and advanced usage of JAX DataLoader.

Basic Usage
----------

Loading Data
~~~~~~~~~~~

The basic way to load data is using the `DataLoader` class with a configuration:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Create a configuration
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       shuffle=True
   )

   # Create a data loader
   loader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in loader:
       # Process your batch here
       pass

Data Formats
~~~~~~~~~~~

JAX DataLoader supports various data formats:

CSV Data
^^^^^^^

.. code-block:: python

   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       target_column="label",
       batch_size=32
   )

JSON Data
^^^^^^^^

.. code-block:: python

   config = DataLoaderConfig(
       data_path="data/train.json",
       loader_type="json",
       target_key="label",
       batch_size=32
   )

Image Data
^^^^^^^^^

.. code-block:: python

   config = DataLoaderConfig(
       data_path="data/images/",
       loader_type="image",
       image_size=(224, 224),
       batch_size=32
   )

Advanced Usage
------------

Data Transformation
~~~~~~~~~~~~~~~~~

You can apply transformations to your data using the `Transform` class:

.. code-block:: python

   from jax_dataloader import Transform

   # Create transformations
   transform = Transform()
   transform.add("random_flip", probability=0.5)
   transform.add("random_rotation", max_angle=30)

   # Add to configuration
   config = DataLoaderConfig(
       data_path="data/images/",
       loader_type="image",
       transform=transform,
       batch_size=32
   )

Memory Management
~~~~~~~~~~~~~~~

JAX DataLoader provides memory management features:

.. code-block:: python

   from jax_dataloader.memory import MemoryManager

   # Create memory manager
   memory_manager = MemoryManager(max_memory=1024**3)  # 1GB

   # Monitor memory usage
   stats = memory_manager.monitor(interval=1.0)
   print(f"Memory usage: {stats['current_usage']}")

Caching
~~~~~~

You can use caching to improve performance:

.. code-block:: python

   from jax_dataloader.memory import Cache

   # Create cache
   cache = Cache(
       max_size=1000,
       eviction_policy="lru"
   )

   # Use cache in data loader
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       cache_size="1GB",
       cache_policy="lru"
   )

   loader = DataLoader(config)

   # Cache will automatically manage frequently accessed data
   for epoch in range(num_epochs):
       for batch_data, batch_labels in loader:
           # Process your batch here
           pass

Progress Tracking
~~~~~~~~~~~~~~~

Track the progress of data loading:

.. code-block:: python

   from jax_dataloader.progress import ProgressTracker

   # Create progress tracker
   tracker = ProgressTracker(
       total=1000,
       update_interval=0.1
   )

   # Use tracker in data loader
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       show_progress=True
   )

   loader = DataLoader(config)

   # Get progress information
   progress = loader.get_progress()
   print(f"Progress: {progress['progress']:.2%}")
   print(f"ETA: {progress['eta']:.2f} seconds")

Multi-GPU Support
~~~~~~~~~~~~~~~

JAX DataLoader supports multi-GPU training:

.. code-block:: python

   import jax

   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32 * jax.device_count(),  # Scale batch size by number of devices
       shuffle=True
   )

   loader = DataLoader(config)

   # Your training function
   @jax.pmap
   def train_step(params, batch):
       # Your training logic here
       pass

Best Practices
-------------

1. **Batch Size Selection**
   - Start with a small batch size and increase based on available memory
   - Use the `calculate_batch_size` utility function for optimal selection

2. **Memory Management**
   - Monitor memory usage with `MemoryManager`
   - Use caching for frequently accessed data
   - Enable memory optimization features when needed

3. **Performance Optimization**
   - Use appropriate number of workers
   - Enable prefetching for better performance
   - Use caching for repeated data access

4. **Error Handling**
   - Always check for data format compatibility
   - Handle memory errors gracefully
   - Use try-except blocks for data loading operations

For more examples and detailed information, see the :doc:`examples` section. 