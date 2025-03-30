Examples
========

This section provides detailed examples of using JAX DataLoader in various scenarios.

Basic Examples
------------

Simple Data Loading
~~~~~~~~~~~~~~~~

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
       print(f"Batch shape: {batch_data.shape}")

Loading from Files
----------------

CSV Data
~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import CSVLoader

   # Create CSV loader
   loader = CSVLoader(
       "data.csv",
       target_column="label",
       feature_columns=["feature1", "feature2"]
   )

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       loader=loader,
       config=config
   )

   # Iterate over batches
   for features, labels in dataloader:
       print(f"Features shape: {features.shape}")
       print(f"Labels shape: {labels.shape}")

JSON Data
~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import JSONLoader

   # Create JSON loader
   loader = JSONLoader(
       "data.json",
       data_key="features",
       label_key="labels"
   )

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       loader=loader,
       config=config
   )

   # Iterate over batches
   for data, labels in dataloader:
       print(f"Data shape: {data.shape}")
       print(f"Labels shape: {labels.shape}")

Image Data
~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import ImageLoader

   # Create image loader
   loader = ImageLoader(
       "image_directory",
       image_size=(224, 224),
       normalize=True
   )

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True,
       num_workers=4
   )

   # Create the dataloader
   dataloader = DataLoader(
       loader=loader,
       config=config
   )

   # Iterate over batches
   for images, labels in dataloader:
       print(f"Images shape: {images.shape}")
       print(f"Labels shape: {labels.shape}")

Advanced Examples
--------------

Multi-GPU Training
~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Get available devices
   devices = jax.devices()
   
   # Create sample data
   data = jnp.arange(10000)
   labels = jnp.arange(10000)

   # Configure for multi-GPU
   config = DataLoaderConfig(
       batch_size=32,
       num_devices=len(devices),
       device_map="auto",
       pin_memory=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Training loop
   for batch_data, batch_labels in dataloader:
       # batch_data and batch_labels are already on the correct devices
       # Your training code here
       pass

Data Augmentation
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp
   import jax.random as random

   # Define augmentation function
   def augment_fn(batch, key):
       # Add random noise
       noise = random.normal(key, batch.shape) * 0.1
       augmented = batch + noise
       
       # Random rotation
       angle = random.uniform(key, minval=-0.1, maxval=0.1)
       augmented = jnp.rot90(augmented, k=int(angle * 10))
       
       return augmented

   # Create sample data
   data = jnp.arange(1000).reshape(100, 10, 10)
   labels = jnp.arange(100)

   # Configure with augmentation
   config = DataLoaderConfig(
       batch_size=32,
       transform=augment_fn,
       transform_key=random.PRNGKey(0)
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Iterate over augmented batches
   for batch_data, batch_labels in dataloader:
       print(f"Augmented batch shape: {batch_data.shape}")

Memory Management
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create large dataset
   data = jnp.arange(1000000)
   labels = jnp.arange(1000000)

   # Configure for memory efficiency
   config = DataLoaderConfig(
       batch_size=32,
       memory_fraction=0.8,
       auto_batch_size=True,
       cache_size=1000,
       num_workers=4
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Enable memory optimization
   dataloader.optimize_memory()

   # Iterate over memory-efficient batches
   for batch_data, batch_labels in dataloader:
       print(f"Batch shape: {batch_data.shape}")
       print(f"Memory usage: {dataloader.memory_manager.get_memory_usage()}")

Progress Tracking
~~~~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp
   import time

   # Create sample data
   data = jnp.arange(1000)
   labels = jnp.arange(1000)

   # Configure with progress tracking
   config = DataLoaderConfig(
       batch_size=32,
       show_progress=True,
       progress_interval=0.1
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Training loop with progress tracking
   start_time = time.time()
   for batch_data, batch_labels in dataloader:
       # Simulate processing time
       time.sleep(0.1)
       
       # Progress bar will show automatically
       print(f"Processing batch...")

   end_time = time.time()
   print(f"Total time: {end_time - start_time:.2f} seconds")

Error Handling
~~~~~~~~~~~~

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.exceptions import DataLoaderError
   import jax.numpy as jnp

   # Create sample data
   data = jnp.arange(1000)
   labels = jnp.arange(1000)

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       error_handling=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

   # Training loop with error handling
   try:
       for batch_data, batch_labels in dataloader:
           try:
               # Your processing code here
               pass
           except Exception as e:
               print(f"Error processing batch: {e}")
               continue
   except DataLoaderError as e:
       print(f"DataLoader error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}") 