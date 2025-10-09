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

   # Follow the installation instructions at https://docs.rapids.ai/install/

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

Metrics Module
~~~~~~~~~~~~~~

Comprehensive evaluation metrics for dimensionality reduction quality:

.. code-block:: python

   from dire_rapids.metrics import evaluate_embedding

   # Comprehensive evaluation
   results = evaluate_embedding(data, layout, labels)

   # Access metrics
   print(f"Stress: {results['local']['stress']:.4f}")
   print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
   print(f"Wasserstein: {results['topology']['metrics']['wass'][0]:.6f}")

**Available metrics:**

* **Distortion**: stress, neighborhood preservation
* **Context**: SVM/kNN classification accuracy preservation
* **Topology**: persistence diagrams, Betti curves, Wasserstein/bottleneck distances

**Persistence backends** (auto-selected): giotto-ph, ripser++, ripser

See :doc:`api/dire_rapids.metrics` for full API reference.

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

ReducerRunner Framework
~~~~~~~~~~~~~~~~~~~~~~~~

General-purpose framework for running any dimensionality reduction algorithm with automatic data loading. See ``benchmarking/dire_rapids_benchmarks.ipynb`` for complete examples.

.. code-block:: python

   from benchmarking.reducer_runner import ReducerRunner
   from dire_rapids import create_dire

   runner = ReducerRunner(reducer_class=create_dire, reducer_kwargs={"n_neighbors": 16})
   result = runner.run("sklearn:blobs")
   result = runner.run("dire:sphere_uniform", dataset_kwargs={"n_features": 10, "n_samples": 1000})

**Data sources:**

* ``sklearn:name`` - sklearn datasets
* ``openml:name`` - OpenML datasets
* ``cytof:name`` - CyTOF datasets
* ``dire:name`` - DiRe geometric datasets (``disk_uniform``, ``sphere_uniform``, ``ellipsoid_uniform``)
* ``file:path`` - Local files

**Reducer comparison:**

.. code-block:: python

   # In notebooks: %run benchmarking/compare_reducers.py
   from benchmarking.compare_reducers import compare_reducers, print_comparison_summary

   results = compare_reducers("sklearn:blobs", metrics=['distortion', 'context'])
   print_comparison_summary(results)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`