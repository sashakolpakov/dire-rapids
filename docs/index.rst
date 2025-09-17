dire-rapids
===========

PyTorch and RAPIDS (cuVS/cuML) accelerated dimensionality reduction.

**dire-rapids** provides high-performance dimensionality reduction using the DiRe algorithm
with multiple backend implementations optimized for different scales and hardware configurations.

Features
--------

* **Multiple backends**: PyTorch, memory-efficient PyTorch, and RAPIDS cuVS
* **Automatic backend selection** based on hardware and dataset characteristics
* **Custom distance metrics** for k-NN computation (string expressions or callables)
* **GPU acceleration** with CUDA support
* **Memory-efficient processing** for large datasets (>100K points)
* **Interactive visualizations** with Plotly
* **Scikit-learn compatible API**

Backends
--------

- **DiRePyTorch**: Standard PyTorch implementation for general use
- **DiRePyTorchMemoryEfficient**: Memory-optimized for large datasets
- **DiReCuVS**: RAPIDS cuVS/cuML accelerated for massive datasets

Installation
------------

Install the base package:

.. code-block:: bash

   pip install dire-rapids

For GPU acceleration with RAPIDS:

.. code-block:: bash

   conda install -c rapidsai -c conda-forge rapids=25.08

Quick Start
-----------

.. code-block:: python

   from dire_rapids import create_dire
   import numpy as np
   
   # Create sample data
   X = np.random.randn(10000, 100)
   
   # Create reducer with automatic backend selection
   reducer = create_dire(n_neighbors=32)

   # Fit and transform data
   embedding = reducer.fit_transform(X)
   
   # Visualize results
   fig = reducer.visualize()
   fig.show()

API Documentation
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from dire_rapids import DiRePyTorch
   import numpy as np
   
   # Create sample data
   X = np.random.randn(5000, 50)
   
   # Create and fit reducer
   reducer = DiRePyTorch(n_neighbors=32, verbose=True)
   embedding = reducer.fit_transform(X)
   
   # Visualize
   fig = reducer.visualize()
   fig.show()

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dire_rapids import DiRePyTorchMemoryEfficient
   
   # For large datasets
   X = np.random.randn(100000, 512)
   
   reducer = DiRePyTorchMemoryEfficient(
       n_neighbors=50,
       use_fp16=True,  # Use half precision for memory efficiency
       verbose=True
   )
   embedding = reducer.fit_transform(X)

GPU Acceleration with RAPIDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dire_rapids import DiReCuVS
   
   # Massive dataset with GPU acceleration
   X = np.random.randn(1000000, 128)
   
   reducer = DiReCuVS(
       use_cuvs=True,
       cuvs_index_type='cagra',  # Best for very large datasets
       n_neighbors=64
   )
   embedding = reducer.fit_transform(X)

Automatic Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dire_rapids import create_dire
   
   # Automatic selection based on hardware
   reducer = create_dire(
       n_neighbors=32,
       memory_efficient=True  # Use memory-efficient variant if needed
   )
   embedding = reducer.fit_transform(X)

Custom Distance Metrics
~~~~~~~~~~~~~~~~~~~~~~~

DiRe supports custom distance metrics for k-nearest neighbor computation:

.. code-block:: python

   # Using L1 (Manhattan) distance
   reducer = DiRePyTorch(
       metric='(x - y).abs().sum(-1)',
       n_neighbors=32
   )
   embedding = reducer.fit_transform(X)

   # Using custom callable metric
   def cosine_distance(x, y):
       return 1 - (x * y).sum(-1) / (
           x.norm(dim=-1, keepdim=True) *
           y.norm(dim=-1, keepdim=True) + 1e-8
       )

   reducer = DiRePyTorch(metric=cosine_distance)
   embedding = reducer.fit_transform(X)

   # Custom metrics work with all backends
   reducer = create_dire(
       metric='(x - y).abs().sum(-1)',  # L1 distance
       memory_efficient=True
   )
   embedding = reducer.fit_transform(X)

**Supported metric types:**

* ``None`` or ``'euclidean'``/``'l2'``: Fast built-in Euclidean (default)
* **String expressions**: Evaluated tensor expressions using ``x`` and ``y``
* **Callable functions**: Custom Python functions taking ``(x, y)`` tensors

Note: Layout forces remain Euclidean regardless of k-NN metric for optimal performance.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`