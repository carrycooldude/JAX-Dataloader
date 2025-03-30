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

DataLoaderConfig
~~~~~~~~~~~~~~

.. autoclass:: jax_dataloader.DataLoaderConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Data Loaders
-----------

CSVLoader
~~~~~~~~

.. autoclass:: jax_dataloader.data.CSVLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

JSONLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.data.JSONLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ImageLoader
~~~~~~~~~~

.. autoclass:: jax_dataloader.data.ImageLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

BaseLoader
~~~~~~~~~

.. autoclass:: jax_dataloader.data.BaseLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Memory Management
---------------

MemoryManager
~~~~~~~~~~~

.. autoclass:: jax_dataloader.memory.MemoryManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Cache
~~~~

.. autoclass:: jax_dataloader.memory.Cache
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Progress Tracking
---------------

ProgressTracker
~~~~~~~~~~~~~

.. autoclass:: jax_dataloader.progress.ProgressTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Data Augmentation
---------------

Transform
~~~~~~~~

.. autoclass:: jax_dataloader.transform.Transform
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Exceptions
---------

DataLoaderError
~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.DataLoaderError
   :members:
   :show-inheritance:

ConfigurationError
~~~~~~~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.ConfigurationError
   :members:
   :show-inheritance:

MemoryError
~~~~~~~~~

.. autoexception:: jax_dataloader.exceptions.MemoryError
   :members:
   :show-inheritance:

Utility Functions
--------------

.. autofunction:: jax_dataloader.utils.get_available_memory
.. autofunction:: jax_dataloader.utils.calculate_batch_size
.. autofunction:: jax_dataloader.utils.get_device_count
.. autofunction:: jax_dataloader.utils.format_size 