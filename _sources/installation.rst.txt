Installation Guide
================

Requirements
-----------

JAX DataLoader requires Python 3.7 or later and the following dependencies:

- JAX >= 0.3.0
- JAXlib >= 0.3.0
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Pillow >= 8.0.0
- psutil >= 5.8.0
- tqdm >= 4.50.0

Installation Methods
------------------

Using pip
~~~~~~~~

The easiest way to install JAX DataLoader is using pip:

.. code-block:: bash

    pip install jax-dataloaders

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features, you can install from source:

.. code-block:: bash

    git clone https://github.com/carrycooldude/JAX-Dataloader.git
    cd JAX-Dataloader
    pip install -e .

Using conda
~~~~~~~~~~

You can also install JAX DataLoader using conda:

.. code-block:: bash

    conda install -c conda-forge jax-dataloaders

Verifying Installation
--------------------

To verify that JAX DataLoader is installed correctly:

.. code-block:: python

    from jax_dataloader import DataLoader
    print(DataLoader.__version__)

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~

1. JAX Installation
   - If you encounter issues with JAX installation, refer to the `JAX installation guide <https://github.com/google/jax#installation>`_.
   - For CUDA support, make sure you have the correct version of CUDA installed.

2. Memory Issues
   - If you encounter memory errors, try reducing the batch size or enabling memory management.
   - Use the `memory_limit` parameter in `DataLoaderConfig` to control memory usage.

3. Multi-GPU Support
   - Ensure JAX is properly configured for multi-GPU usage.
   - Check that your batch size is compatible with the number of devices.

Getting Help
~~~~~~~~~~~

If you encounter any issues:

1. Check the `GitHub issues <https://github.com/carrycooldude/JAX-Dataloader/issues>`_ to see if your problem has been reported.
2. If not, create a new issue with details about your problem.
3. Join our `Discord community <https://discord.gg/your-server>`_ for real-time support. 