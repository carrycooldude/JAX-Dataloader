Usage Guide
===========

This guide will help you get started with JAX DataLoader and show you how to use its various features effectively.

Basic Usage
----------

The most basic usage of JAX DataLoader involves creating a DataLoader instance with your data and configuration:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create sample data
   data = jnp.arange(1000)
   labels = jnp.arange(1000)

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True,
       drop_last=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       # Process your batch
       print(f"Batch shape: {batch_data.shape}")

Configuration Options
-------------------

The DataLoaderConfig class provides various options to customize your data loading:

.. code-block:: python

   config = DataLoaderConfig(
       batch_size=32,          # Size of each batch
       shuffle=True,           # Whether to shuffle data
       drop_last=False,        # Whether to drop the last incomplete batch
       num_workers=4,          # Number of worker processes
       pin_memory=True,        # Whether to pin memory for faster GPU transfer
       prefetch_factor=2,      # Number of batches to prefetch
       persistent_workers=True # Whether to keep workers alive between epochs
   )

Memory Management
---------------

JAX DataLoader provides several features for efficient memory management:

.. code-block:: python

   config = DataLoaderConfig(
       batch_size=32,
       memory_fraction=0.8,    # Maximum fraction of available memory to use
       auto_batch_size=True,   # Automatically adjust batch size based on memory
       cache_size=1000         # Number of batches to cache
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

   # Enable memory optimization
   dataloader.optimize_memory()

Multi-GPU Support
---------------

To use multiple GPUs, you can configure the DataLoader to distribute data across devices:

.. code-block:: python

   import jax
   from jax_dataloader import DataLoader, DataLoaderConfig

   # Get available devices
   devices = jax.devices()
   
   config = DataLoaderConfig(
       batch_size=32,
       num_devices=len(devices),  # Number of devices to use
       device_map="auto"          # Automatic device mapping
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

   # Data will be automatically distributed across devices
   for batch in dataloader:
       # batch will be a tuple of (data, device_id)
       data, device_id = batch

Progress Tracking
----------------

You can track the progress of data loading using the built-in progress bar:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   data = jnp.arange(1000)
   config = DataLoaderConfig(
       batch_size=32,
       show_progress=True,     # Enable progress bar
       progress_interval=0.1   # Update interval in seconds
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

   for batch in dataloader:
       # Progress bar will show automatically
       pass

Data Augmentation
----------------

JAX DataLoader supports data augmentation through the transform system:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp
   import jax.random as random

   def augment_fn(batch, key):
       # Example augmentation: add random noise
       noise = random.normal(key, batch.shape) * 0.1
       return batch + noise

   config = DataLoaderConfig(
       batch_size=32,
       transform=augment_fn,    # Apply augmentation function
       transform_key=random.PRNGKey(0)  # Random key for augmentation
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

Loading Different Data Formats
----------------------------

JAX DataLoader supports various data formats:

CSV Files:
~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import CSVLoader

   loader = CSVLoader("data.csv")
   config = DataLoaderConfig(batch_size=32)
   dataloader = DataLoader(loader=loader, config=config)

JSON Files:
~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import JSONLoader

   loader = JSONLoader("data.json")
   config = DataLoaderConfig(batch_size=32)
   dataloader = DataLoader(loader=loader, config=config)

Image Files:
~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import ImageLoader

   loader = ImageLoader("image_directory")
   config = DataLoaderConfig(
       batch_size=32,
       image_size=(224, 224)  # Resize images to 224x224
   )
   dataloader = DataLoader(loader=loader, config=config)

Best Practices
-------------

1. **Batch Size Selection**
   - Start with a small batch size and increase based on available memory
   - Use auto_batch_size=True for automatic optimization
   - Consider using gradient accumulation for large models

2. **Memory Management**
   - Enable pin_memory=True when using GPU
   - Use memory_fraction to limit memory usage
   - Enable caching for frequently accessed data

3. **Performance Optimization**
   - Use num_workers > 0 for parallel data loading
   - Enable persistent_workers=True for better performance
   - Use prefetch_factor to overlap data loading with computation

4. **Error Handling**
   - Always wrap data loading in try-except blocks
   - Use the built-in error handling features
   - Monitor memory usage and adjust configuration accordingly

For more advanced usage and examples, check out the :doc:`examples` guide. 