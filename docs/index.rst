Welcome to JAX DataLoader's documentation!
=======================================

JAX DataLoader is a high-performance data loading library for JAX that provides efficient data loading, preprocessing, and augmentation capabilities.

Installation
-----------

.. code-block:: bash

   pip install jax-dataloaders

Quick Start
----------

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig

   # Create a configuration
   config = DataLoaderConfig(
       batch_size=32,
       shuffle=True,
       num_workers=4
   )

   # Create a data loader
   loader = DataLoader(config)

   # Iterate over batches
   for batch in loader:
       # Process your batch
       pass

Features
--------

* Efficient data loading with JAX
* Support for various data formats (CSV, JSON, Images)
* Data augmentation and preprocessing
* Progress tracking and monitoring
* Memory management and caching
* Multi-GPU support
* Memory monitoring and optimization

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   tutorials
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`