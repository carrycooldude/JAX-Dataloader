Installation Guide
=================

This guide will help you install JAX DataLoader and its dependencies.

Requirements
-----------

* Python 3.7 or higher
* JAX and JAXlib
* NumPy
* Optional: PyTorch (for some data format support)

Basic Installation
-----------------

The simplest way to install JAX DataLoader is using pip:

.. code-block:: bash

   pip install jax-dataloaders

This will install the latest stable version from PyPI along with its core dependencies.

Development Installation
----------------------

If you want to contribute to the project or need the latest features, you can install from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/carrycooldude/JAX-Dataloader.git
   cd JAX-Dataloader

   # Install in editable mode
   pip install -e .

   # Install development dependencies
   pip install -e ".[dev]"

Installing with Optional Dependencies
-----------------------------------

JAX DataLoader has several optional dependencies that you can install based on your needs:

* For CSV support:
  .. code-block:: bash

     pip install "jax-dataloaders[csv]"

* For JSON support:
  .. code-block:: bash

     pip install "jax-dataloaders[json]"

* For image support:
  .. code-block:: bash

     pip install "jax-dataloaders[image]"

* For all optional dependencies:
  .. code-block:: bash

     pip install "jax-dataloaders[all]"

GPU Support
----------

To use JAX DataLoader with GPU support, you'll need to install the appropriate JAX version for your CUDA version. Follow the `official JAX installation guide <https://github.com/google/jax#installation>`_ for detailed instructions.

For example, for CUDA 11.8:

.. code-block:: bash

   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Verifying Installation
--------------------

You can verify your installation by running a simple test:

.. code-block:: python

   from jax_dataloader import DataLoader, DataLoaderConfig
   import jax.numpy as jnp

   # Create test data
   data = jnp.arange(100)
   config = DataLoaderConfig(batch_size=10)
   dataloader = DataLoader(data=data, config=config)

   # Try iterating
   for batch in dataloader:
       print(f"Batch shape: {batch.shape}")
       break

If you see the batch shape printed without any errors, your installation is successful!

Troubleshooting
--------------

Common issues and their solutions:

1. **ImportError: No module named 'jax'**
   Make sure you have JAX installed correctly. Try reinstalling JAX following the official guide.

2. **CUDA errors**
   Ensure your CUDA version matches the JAX version you installed. Check the JAX installation guide for compatibility.

3. **Memory issues**
   If you encounter memory errors, try reducing the batch size or enabling memory optimization features.

For more help, please check the :doc:`usage` guide or open an issue on the `GitHub repository <https://github.com/carrycooldude/JAX-Dataloader/issues>`_. 