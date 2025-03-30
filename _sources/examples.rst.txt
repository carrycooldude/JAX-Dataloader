Examples
========

Basic Examples
-------------

Simple Data Loading
~~~~~~~~~~~~~~~~~

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

Loading from Files
-----------------

CSV Data
~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Create a configuration for CSV data
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       shuffle=True,
       target_column="label"
   )

   # Create a data loader
   loader = DataLoader(config)

   # Get metadata about the dataset
   metadata = loader.get_metadata()
   print(f"Number of samples: {metadata['num_samples']}")
   print(f"Number of features: {metadata['num_features']}")
   print(f"Feature names: {metadata['feature_names']}")

   # Iterate over batches
   for batch_data, batch_labels in loader:
       # Process your batch here
       pass

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

   # Create a configuration for image data
   config = DataLoaderConfig(
       data_path="data/images/",
       loader_type="image",
       batch_size=16,
       shuffle=True,
       image_size=(224, 224)
   )

   # Create a data loader
   loader = DataLoader(config)

   # Iterate over batches
   for batch_images, batch_labels in loader:
       # Process your batch here
       pass

Advanced Examples
--------------

Multi-GPU Training
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax

   # Create a configuration
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32 * jax.device_count(),  # Scale batch size by number of devices
       shuffle=True
   )

   # Create a data loader
   loader = DataLoader(config)

   # Your training function
   @jax.pmap
   def train_step(params, batch):
       # Your training logic here
       pass

Data Augmentation
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig, Transform

   # Create transformations
   transform = Transform()
   transform.add("random_flip", probability=0.5)
   transform.add("random_rotation", max_angle=30)
   transform.add("random_brightness", max_delta=0.2)

   # Create a configuration with transformations
   config = DataLoaderConfig(
       data_path="data/images/",
       loader_type="image",
       batch_size=16,
       shuffle=True,
       transform=transform
   )

   # Create a data loader
   loader = DataLoader(config)

Memory Management
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Create a configuration with memory management
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       shuffle=True,
       memory_limit="4GB",  # Limit memory usage
       cache_size="1GB"     # Set cache size
   )

   # Create a data loader
   loader = DataLoader(config)

   # Monitor memory usage
   memory_stats = loader.get_memory_usage()
   print(f"Current memory usage: {memory_stats['current_usage']}")
   print(f"Peak memory usage: {memory_stats['peak_usage']}")

Progress Tracking
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Create a configuration with progress tracking
   config = DataLoaderConfig(
       data_path="data/train.csv",
       loader_type="csv",
       batch_size=32,
       shuffle=True,
       show_progress=True  # Enable progress tracking
   )

   # Create a data loader
   loader = DataLoader(config)

   # Get progress information
   progress = loader.get_progress()
   print(f"Current batch: {progress['current_batch']}")
   print(f"Total batches: {progress['total_batches']}")
   print(f"Progress: {progress['progress']:.2%}")
   print(f"ETA: {progress['eta']:.2f} seconds")

Error Handling
~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.exceptions import DataLoaderError, ConfigurationError

   try:
       # Create a configuration
       config = DataLoaderConfig(
           data_path="nonexistent.csv",
           loader_type="csv",
           batch_size=32
       )

       # Create a data loader
       loader = DataLoader(config)

   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   except DataLoaderError as e:
       print(f"Data loader error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

For more examples and use cases, check out the `GitHub repository <https://github.com/carrycooldude/JAX-Dataloader/tree/main/examples>`_. 