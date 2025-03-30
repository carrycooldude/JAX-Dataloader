Tutorials
=========

This section provides step-by-step tutorials for common use cases with JAX DataLoader.

Getting Started with Image Classification
---------------------------------------

In this tutorial, we'll create a complete image classification pipeline using JAX DataLoader.

1. First, let's set up our environment:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import ImageLoader

2. Create a data loader for our image dataset:

.. code-block:: python

   # Create image loader
   loader = ImageLoader(
       "path/to/image/dataset",
       image_size=(224, 224),
       normalize=True,
       augment=True  # Enable built-in augmentations
   )

   # Configure the dataloader
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )

   # Create the dataloader
   dataloader = DataLoader(
       loader=loader,
       config=config
   )

3. Define a simple model:

.. code-block:: python

   def model(params, x):
       # Simple CNN
       x = jax.nn.relu(jnp.dot(x, params['w1']) + params['b1'])
       x = jax.nn.relu(jnp.dot(x, params['w2']) + params['b2'])
       return jnp.dot(x, params['w3']) + params['b3']

4. Training loop:

.. code-block:: python

   # Initialize parameters
   params = {
       'w1': jax.random.normal(jax.random.PRNGKey(0), (224*224*3, 128)),
       'b1': jnp.zeros(128),
       'w2': jax.random.normal(jax.random.PRNGKey(1), (128, 64)),
       'b2': jnp.zeros(64),
       'w3': jax.random.normal(jax.random.PRNGKey(2), (64, 10)),
       'b3': jnp.zeros(10)
   }

   # Training loop
   for epoch in range(num_epochs):
       for batch_data, batch_labels in dataloader:
           # Forward pass
           predictions = model(params, batch_data)
           
           # Compute loss
           loss = jnp.mean((predictions - batch_labels) ** 2)
           
           # Backward pass (using JAX's grad)
           grads = jax.grad(lambda p: jnp.mean((model(p, batch_data) - batch_labels) ** 2))(params)
           
           # Update parameters
           params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)

Large-Scale Data Processing
-------------------------

This tutorial demonstrates how to handle large datasets efficiently.

1. Set up memory-efficient data loading:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.data import CSVLoader

   # Create CSV loader with chunking
   loader = CSVLoader(
       "large_dataset.csv",
       chunk_size=10000,  # Process in chunks
       target_column="target"
   )

   # Configure for memory efficiency
   config = DataLoaderConfig(
       batch_size=32,
       memory_fraction=0.8,
       auto_batch_size=True,
       cache_size=1000,
       num_workers=4
   )

   dataloader = DataLoader(
       loader=loader,
       config=config
   )

2. Process data in batches:

.. code-block:: python

   # Enable memory optimization
   dataloader.optimize_memory()

   # Process data
   for batch_data, batch_labels in dataloader:
       # Process batch
       process_batch(batch_data, batch_labels)
       
       # Monitor memory usage
       print(f"Memory usage: {dataloader.memory_manager.get_memory_usage()}")

Multi-GPU Training
----------------

Learn how to distribute training across multiple GPUs.

1. Set up multi-GPU configuration:

.. code-block:: python

   import jax
   from jax_dataloader import DataLoader, DataLoaderConfig

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

   dataloader = DataLoader(
       data=data,
       labels=labels,
       config=config
   )

2. Implement distributed training:

.. code-block:: python

   # Training loop
   for batch_data, batch_labels in dataloader:
       # batch_data and batch_labels are already on the correct devices
       data, device_id = batch_data
       
       # Train on specific device
       with jax.device(devices[device_id]):
           # Your training code here
           train_step(data, batch_labels)

Custom Data Augmentation
----------------------

Learn how to create custom data augmentation pipelines.

1. Define augmentation functions:

.. code-block:: python

   import jax.random as random
   import jax.numpy as jnp

   def custom_augment(batch, key):
       # Split key for multiple augmentations
       key1, key2, key3 = random.split(key, 3)
       
       # Add noise
       noise = random.normal(key1, batch.shape) * 0.1
       augmented = batch + noise
       
       # Random rotation
       angle = random.uniform(key2, minval=-0.1, maxval=0.1)
       augmented = jnp.rot90(augmented, k=int(angle * 10))
       
       # Random flip
       if random.uniform(key3) > 0.5:
           augmented = jnp.flip(augmented, axis=1)
       
       return augmented

2. Apply custom augmentations:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Configure with custom augmentation
   config = DataLoaderConfig(
       batch_size=32,
       transform=custom_augment,
       transform_key=random.PRNGKey(0)
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

3. Use in training:

.. code-block:: python

   for batch_data, batch_labels in dataloader:
       # batch_data is already augmented
       train_step(batch_data, batch_labels)

Advanced Caching Strategies
-------------------------

Learn how to optimize data loading with advanced caching.

1. Set up caching:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   from jax_dataloader.memory import Cache

   # Create cache
   cache = Cache(
       max_size=1000,  # Maximum number of batches to cache
       eviction_policy="lru"  # Least Recently Used
   )

   # Configure dataloader with cache
   config = DataLoaderConfig(
       batch_size=32,
       cache=cache,
       cache_hits=True  # Track cache hits
   )

   dataloader = DataLoader(
       data=data,
       config=config
   )

2. Monitor cache performance:

.. code-block:: python

   for batch_data, batch_labels in dataloader:
       # Process batch
       process_batch(batch_data, batch_labels)
       
       # Print cache statistics
       print(f"Cache hits: {dataloader.cache.hits}")
       print(f"Cache misses: {dataloader.cache.misses}")
       print(f"Hit rate: {dataloader.cache.hit_rate}") 