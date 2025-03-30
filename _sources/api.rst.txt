API Reference
=============

Core Classes
-----------

DataLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.DataLoader
   :no-index:
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
~~~~~~~~~~~~~~~

.. autoclass:: jax_dataloader.DataLoaderConfig
   :no-index:
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
   :no-index:
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
   :no-index:
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
   :no-index:
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
              "rotation": [-30, 30],
              "flip": True,
              "brightness": [0.8, 1.2]
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
   :no-index:
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
~~~~~~~~~~~~

.. autoclass:: jax_dataloader.memory.MemoryManager
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic memory management:

   .. code-block:: python

      manager = MemoryManager(max_memory=1024**3)  # 1GB

   Advanced memory management with monitoring:

   .. code-block:: python

      manager = MemoryManager(max_memory=1024**3)
      stats = manager.monitor(interval=1.0)
      print(f"Memory usage: {stats['current_usage']}")

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: allocate
   .. automethod:: deallocate
   .. automethod:: free
   .. automethod:: get_usage
   .. automethod:: cleanup
   .. automethod:: monitor

Cache
~~~~~

.. autoclass:: jax_dataloader.memory.Cache
   :no-index:
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
   :no-index:
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
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   Basic transformation:

   .. code-block:: python

      transform = Transform()
      transform.add(lambda x: x * 2)

   Advanced transformation with chaining:

   .. code-block:: python

      transform = Transform()
      transform.add(lambda x: x * 2)
      transform.add(lambda x: x + 1)
      transform.add(lambda x: jnp.clip(x, 0, 1))

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: add
   .. automethod:: apply
   .. automethod:: compose
   .. automethod:: chain

Exceptions
---------

DataLoaderError
~~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.DataLoaderError
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

ConfigurationError
~~~~~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.ConfigurationError
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

MemoryError
~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.MemoryError
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
---------------

.. autofunction:: jax_dataloader.utils.get_memory_usage
   :no-index:

.. autofunction:: jax_dataloader.utils.format_size
   :no-index:

.. autofunction:: jax_dataloader.utils.get_gpu_memory_usage
   :no-index: 