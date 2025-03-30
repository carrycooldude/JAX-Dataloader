Examples
========

Basic Examples
-------------

Simple Data Loading
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       target_column="label",
       feature_columns=["feature1", "feature2"],
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Labels shape: {batch_labels.shape}")

Loading from Files
-----------------

CSV Data
~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       target_column="label",
       feature_columns=["feature1", "feature2"],
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Labels shape: {batch_labels.shape}")

JSON Data
~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create configuration
   config = DataLoaderConfig(
       loader_type="json",
       data_path="data.json",
       data_key="features",
       label_key="labels",
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Labels shape: {batch_labels.shape}")

Image Data
~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create configuration
   config = DataLoaderConfig(
       loader_type="image",
       data_path="image_directory",
       image_size=(224, 224),
       normalize=True,
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Labels shape: {batch_labels.shape}")

Advanced Examples
--------------

Multi-GPU Training
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax
   import jax.numpy as jnp

   # Get available devices
   devices = jax.devices()

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       batch_size=32 * len(devices),  # Scale batch size with number of devices
       shuffle=True,
       multi_gpu=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Number of devices: {len(devices)}")

Data Augmentation
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.transform import Transform
   import jax.numpy as jnp

   # Create transform
   transform = Transform()
   transform.add(lambda x: x * 2)  # Example transform
   transform.add(lambda x: jnp.clip(x, 0, 1))  # Clip values

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       transform=transform,
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Transformed batch shape: {batch_data.shape}")

Memory Management
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.memory import MemoryManager
   from jax_dataloader.utils import calculate_batch_size

   # Create memory manager
   memory_manager = MemoryManager(max_memory=1024**3)  # 1GB

   # Calculate optimal batch size
   batch_size = calculate_batch_size(
       total_size=10000,
       max_memory=1024**3,
       sample_size=1000
   )

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       batch_size=batch_size,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Monitor memory usage
   stats = memory_manager.monitor(interval=1.0)
   print(f"Memory usage: {stats['current_usage']}")

Progress Tracking
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.progress import ProgressTracker
   import jax.numpy as jnp

   # Create progress tracker
   tracker = ProgressTracker(
       total=1000,
       update_interval=0.1,
       show_eta=True
   )

   # Create configuration
   config = DataLoaderConfig(
       loader_type="csv",
       data_path="data.csv",
       progress_tracker=tracker,
       batch_size=32,
       shuffle=True
   )

   # Create data loader
   dataloader = DataLoader(config)

   # Iterate over batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")

   # Print progress statistics
   print(f"Progress: {tracker.get_progress():.1%}")
   print(f"ETA: {tracker.get_eta():.1f} seconds")

Error Handling
~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.exceptions import DataLoaderError

   try:
       config = DataLoaderConfig(
           loader_type="invalid",
           data_path="data.json"
       )
       dataloader = DataLoader(config)
   except ValueError as e:
       print(f"Error: {e}")

For more examples and use cases, check out the `GitHub repository <https://github.com/carrycooldude/JAX-Dataloader/tree/main/examples>`_. 