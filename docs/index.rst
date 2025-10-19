dire-rapids
===========

PyTorch and RAPIDS accelerated dimensionality reduction.

Features
--------

* Multiple backends: PyTorch, memory-efficient, RAPIDS cuVS
* Automatic backend selection
* Custom distance metrics for k-NN
* GPU acceleration with CUDA
* Memory-efficient processing (>100K points)
* WebGL visualization (100K+ points)
* Scikit-learn compatible API

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
   
   # Visualize (uses WebGL for performance)
   fig = reducer.visualize(max_points=10000)
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

Evaluation metrics for dimensionality reduction quality:

.. code-block:: python

   from dire_rapids.metrics import evaluate_embedding

   # Full evaluation
   results = evaluate_embedding(data, layout, labels, compute_topology=True)

   print(f"Stress: {results['local']['stress']:.4f}")
   print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
   print(f"DTW β₀: {results['topology']['metrics']['dtw_beta0']:.6f}")
   print(f"DTW β₁: {results['topology']['metrics']['dtw_beta1']:.6f}")

**Metrics:**

* **Distortion**: stress, neighborhood preservation
* **Context**: SVM/kNN classification accuracy
* **Topology**: DTW distances between Betti curves (β₀, β₁) via kNN-atlas approach and Hodge Laplacians

See :doc:`api/dire_rapids.metrics` for full API reference.

Custom Distance Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Custom metrics for k-nearest neighbor computation:

.. code-block:: python

   # L1 distance
   reducer = DiRePyTorch(metric='(x - y).abs().sum(-1)', n_neighbors=32)
   embedding = reducer.fit_transform(X)

   # Cosine distance
   def cosine_distance(x, y):
       return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)

   reducer = DiRePyTorch(metric=cosine_distance)
   embedding = reducer.fit_transform(X)

**Metric types:** ``None``/``'euclidean'``/``'l2'`` (default), string expressions, callable functions

Note: Layout forces use Euclidean distance regardless of k-NN metric.

ReducerRunner Framework
~~~~~~~~~~~~~~~~~~~~~~~~

Framework for running sklearn-compatible reducers with automatic data loading and metrics.

.. code-block:: python

   from dire_rapids.dire_pytorch import ReducerRunner, ReducerConfig
   from dire_rapids import create_dire

   config = ReducerConfig(
       name="DiRe",
       reducer_class=create_dire,
       reducer_kwargs={"n_neighbors": 16},
       visualize=True
   )

   runner = ReducerRunner(config=config)
   result = runner.run("sklearn:blobs")
   result = runner.run("cytof:levine32")

**Data sources:** ``sklearn:name``, ``openml:name``, ``cytof:name``, ``dire:name``, ``file:path``

**Compare reducers:**

.. code-block:: python

   from benchmarking.compare_reducers import compare_reducers

   results = compare_reducers("sklearn:digits", metrics=['distortion', 'context', 'topology'])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`