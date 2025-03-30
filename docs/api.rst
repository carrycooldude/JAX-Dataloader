API Reference
=============

This section provides detailed documentation for the JAX DataLoader API.

Core Classes
-----------

DataLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.DataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __iter__, __next__

   .. rubric:: Examples

   Basic usage:

   .. code-block:: python

      from jax_dataloader import DataLoader, DataLoaderConfig
      import jax.numpy as jnp

      # Create sample data
      data = jnp.arange(1000)
      labels = jnp.arange(1000)

      # Configure the dataloader
      config = DataLoaderConfig(
          batch_size=32,
          shuffle=True
      )

      # Create the dataloader
      dataloader = DataLoader(
          data=data,
          labels=labels,
          config=config
      )

      # Iterate over batches
      for batch_data, batch_labels in dataloader:
          print(f"Batch shape: {batch_data.shape}")

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: __iter__
   .. automethod:: __next__
   .. automethod:: optimize_memory
   .. automethod:: get_memory_usage
   .. automethod:: reset
   .. automethod:: get_progress

DataLoaderConfig
~~~~~~~~~~~~~~

.. autoclass:: jax_dataloader.DataLoaderConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic configuration:

   .. code-block:: python

      config = DataLoaderConfig(
          batch_size=32,
          shuffle=True,
          drop_last=True,
          num_workers=4,
          pin_memory=True
      )

   Advanced configuration with memory management:

   .. code-block:: python

      config = DataLoaderConfig(
          batch_size=32,
          memory_fraction=0.8,
          auto_batch_size=True,
          cache_size=1000,
          num_workers=4,
          prefetch_factor=2,
          persistent_workers=True
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: validate
   .. automethod:: to_dict
   .. automethod:: from_dict

Data Loaders
-----------

CSVLoader
~~~~~~~~

.. autoclass:: jax_dataloader.data.CSVLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic CSV loading:

   .. code-block:: python

      loader = CSVLoader(
          "data.csv",
          target_column="label",
          feature_columns=["feature1", "feature2"]
      )

   Advanced CSV loading with chunking:

   .. code-block:: python

      loader = CSVLoader(
          "large_dataset.csv",
          chunk_size=10000,
          target_column="target",
          feature_columns=["feature1", "feature2"],
          dtype=jnp.float32
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: load
   .. automethod:: get_chunk
   .. automethod:: get_metadata

JSONLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.data.JSONLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic JSON loading:

   .. code-block:: python

      loader = JSONLoader(
          "data.json",
          data_key="features",
          label_key="labels"
      )

   Advanced JSON loading with preprocessing:

   .. code-block:: python

      loader = JSONLoader(
          "data.json",
          data_key="features",
          label_key="labels",
          preprocess_fn=lambda x: x / 255.0,
          dtype=jnp.float32
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: load
   .. automethod:: preprocess
   .. automethod:: get_metadata

ImageLoader
~~~~~~~~~~

.. autoclass:: jax_dataloader.data.ImageLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic image loading:

   .. code-block:: python

      loader = ImageLoader(
          "image_directory",
          image_size=(224, 224),
          normalize=True
      )

   Advanced image loading with augmentation:

   .. code-block:: python

      loader = ImageLoader(
          "image_directory",
          image_size=(224, 224),
          normalize=True,
          augment=True,
          augment_options={
              "rotation": (-10, 10),
              "flip": True,
              "brightness": (0.8, 1.2)
          }
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: load
   .. automethod:: preprocess
   .. automethod:: augment
   .. automethod:: get_metadata

BaseLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.data.BaseLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: load
   .. automethod:: preprocess
   .. automethod:: get_metadata

Memory Management
---------------

MemoryManager
~~~~~~~~~~~

.. autoclass:: jax_dataloader.memory.MemoryManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic memory management:

   .. code-block:: python

      manager = MemoryManager(
          max_memory=0.8,  # 80% of available memory
          auto_cleanup=True
      )

   Advanced memory management with monitoring:

   .. code-block:: python

      manager = MemoryManager(
          max_memory=0.8,
          auto_cleanup=True,
          monitor_interval=1.0,
          warning_threshold=0.9
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: allocate
   .. automethod:: free
   .. automethod:: get_usage
   .. automethod:: cleanup
   .. automethod:: monitor

Cache
~~~~

.. autoclass:: jax_dataloader.memory.Cache
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic caching:

   .. code-block:: python

      cache = Cache(
          max_size=1000,
          eviction_policy="lru"
      )

   Advanced caching with statistics:

   .. code-block:: python

      cache = Cache(
          max_size=1000,
          eviction_policy="lru",
          track_stats=True,
          max_age=3600  # 1 hour
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: get
   .. automethod:: put
   .. automethod:: clear
   .. automethod:: get_stats
   .. automethod:: evict

Progress Tracking
---------------

ProgressTracker
~~~~~~~~~~~~~

.. autoclass:: jax_dataloader.progress.ProgressTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic progress tracking:

   .. code-block:: python

      tracker = ProgressTracker(
          total=1000,
          update_interval=0.1
      )

   Advanced progress tracking with callbacks:

   .. code-block:: python

      def on_update(progress):
          print(f"Progress: {progress:.1%}")

      tracker = ProgressTracker(
          total=1000,
          update_interval=0.1,
          callbacks=[on_update],
          show_eta=True
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: update
   .. automethod:: reset
   .. automethod:: get_progress
   .. automethod:: get_eta

Data Augmentation
---------------

Transform
~~~~~~~~

.. autoclass:: jax_dataloader.transform.Transform
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic transformation:

   .. code-block:: python

      transform = Transform(
          fn=lambda x: x + 1,
          key=random.PRNGKey(0)
      )

   Advanced transformation with multiple operations:

   .. code-block:: python

      def augment_fn(batch, key):
          key1, key2 = random.split(key)
          noise = random.normal(key1, batch.shape) * 0.1
          angle = random.uniform(key2, minval=-0.1, maxval=0.1)
          return jnp.rot90(batch + noise, k=int(angle * 10))

      transform = Transform(
          fn=augment_fn,
          key=random.PRNGKey(0)
      )

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: apply
   .. automethod:: compose
   .. automethod:: chain

Exceptions
---------

DataLoaderError
~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.DataLoaderError
   :members:
   :show-inheritance:

   .. rubric:: Examples

   .. code-block:: python

      try:
          dataloader = DataLoader(data=None)
      except DataLoaderError as e:
          print(f"Error: {e}")

ConfigurationError
~~~~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.ConfigurationError
   :members:
   :show-inheritance:

   .. rubric:: Examples

   .. code-block:: python

      try:
          config = DataLoaderConfig(batch_size=-1)
      except ConfigurationError as e:
          print(f"Error: {e}")

MemoryError
~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.MemoryError
   :members:
   :show-inheritance:

   .. rubric:: Examples

   .. code-block:: python

      try:
          dataloader = DataLoader(data=large_data)
          dataloader.optimize_memory()
      except MemoryError as e:
          print(f"Error: {e}")

Utility Functions
--------------

.. autofunction:: jax_dataloader.utils.get_available_memory
   :noindex:

   .. rubric:: Examples

   .. code-block:: python

      memory = get_available_memory()
      print(f"Available memory: {memory / 1024**3:.2f} GB")

.. autofunction:: jax_dataloader.utils.calculate_batch_size
   :noindex:

   .. rubric:: Examples

   .. code-block:: python

      batch_size = calculate_batch_size(
          total_size=10000,
          memory_fraction=0.8
      )
      print(f"Recommended batch size: {batch_size}")

.. autofunction:: jax_dataloader.utils.get_device_count
   :noindex:

   .. rubric:: Examples

   .. code-block:: python

      num_devices = get_device_count()
      print(f"Number of available devices: {num_devices}")

.. autofunction:: jax_dataloader.utils.format_size
   :noindex:

   .. rubric:: Examples

   .. code-block:: python

      size = format_size(1024**3)
      print(f"Formatted size: {size}")  # "1.00 GB" 