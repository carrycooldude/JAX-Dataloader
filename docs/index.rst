Welcome to JAX DataLoader's documentation!
=======================================

JAX DataLoader is a high-performance data loading library for JAX that provides efficient data loading and preprocessing capabilities. It is designed to be simple, fast, and memory-efficient, making it perfect for deep learning and data science workflows.

Features
--------

* **High Performance**: Optimized data loading with minimal overhead
* **Memory Efficient**: Smart memory management and data streaming
* **Flexible**: Support for various data formats (CSV, JSON, Images)
* **Easy to Use**: Simple API with familiar interface
* **Type Safe**: Full type hints and static type checking
* **Extensible**: Easy to add custom data loaders

Quick Start
----------

.. code-block:: python

   from jax_dataloader import DataLoader
   import jax.numpy as jnp

   # Create a simple dataset
   data = jnp.array([1, 2, 3, 4, 5])
   dataset = DataLoader(data, batch_size=2)

   # Iterate over batches
   for batch in dataset:
       print(batch)

Installation
-----------

.. code-block:: bash

   pip install jax-dataloaders

For development installation:

.. code-block:: bash

   git clone https://github.com/carrycooldude/JAX-Dataloader.git
   cd JAX-Dataloader
   pip install -e ".[dev]"

Documentation Contents
--------------------

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

Getting Help
-----------

If you encounter any issues or have questions:

* Open an issue on `GitHub <https://github.com/carrycooldude/JAX-Dataloader/issues>`_
* Check the `examples <examples.html>`_ for common use cases
* Join our `Discussions <https://github.com/carrycooldude/JAX-Dataloader/discussions>`_

Contributing
-----------

We welcome contributions! Please see our `Contributing Guide <https://github.com/carrycooldude/JAX-Dataloader/blob/main/CONTRIBUTING.md>`_ for details.