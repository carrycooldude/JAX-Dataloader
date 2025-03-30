Welcome to JAX DataLoader's documentation!
======================================

JAX DataLoader is a high-performance data loading library for JAX applications, providing efficient data loading, batching, and preprocessing capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   tutorials
   api
   examples
   changelog

Features
--------

* Efficient data loading with automatic batching
* Memory management and optimization
* Multi-GPU support
* Progress tracking
* Automatic batch size tuning
* Support for various data formats (CSV, JSON, Images)
* Data augmentation capabilities
* Caching system for improved performance

Installation
-----------

You can install JAX DataLoader using pip:

.. code-block:: bash

   pip install jax-dataloaders

For development installation:

.. code-block:: bash

   git clone https://github.com/carrycooldude/JAX-Dataloader.git
   cd JAX-Dataloader
   pip install -e .

Quick Start
----------

Here's a simple example of how to use JAX DataLoader:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create some sample data
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

For more detailed examples and usage instructions, see the :doc:`usage` guide.

Documentation Sections
--------------------

* :doc:`installation` - Installation guide and requirements
* :doc:`usage` - Usage guide with examples
* :doc:`tutorials` - Step-by-step tutorials for common use cases
* :doc:`api` - Complete API reference
* :doc:`examples` - Detailed examples for various scenarios
* :doc:`changelog` - Version history and changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`