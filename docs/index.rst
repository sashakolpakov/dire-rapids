dire-rapids
===========

PyTorch and RAPIDS accelerated dimensionality reduction.

Features
--------

* Multiple reducer implementations: PyTorch, memory-efficient, RAPIDS cuVS
* Automatic backend selection with explicit k-NN engine overrides
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

``backend`` controls which reducer implementation is constructed.
``knn_backend`` controls the k-NN engine used inside that reducer:
``'auto'``, ``'pytorch'``, ``'pykeops'``, or ``'cuvs'``. Manual k-NN engine
requests are strict and raise if the requested engine cannot run.

Installation
------------

Install the base package:

.. code-block:: bash

   python -m pip install "dire-rapids==0.3.2"

Install optional k-NN engines:

.. code-block:: bash

   # PyKeOps k-NN engine
   python -m pip install "dire-rapids[keops]==0.3.2"

   # CUDA CuPy support
   python -m pip install "dire-rapids[cuda]==0.3.2"

For GPU acceleration with RAPIDS:

Use a clean virtual environment. The ``rapids`` extra installs cuML/cuVS/cuDF
from the NVIDIA index and PyTorch from the matching CUDA wheel index.

.. code-block:: bash

   python -m pip install \
     --extra-index-url https://pypi.nvidia.com \
     --extra-index-url https://download.pytorch.org/whl/cu128 \
     "dire-rapids[rapids,keops]==0.3.2"

For development from a clone:

.. code-block:: bash

   git clone https://github.com/sashakolpakov/dire-rapids.git
   cd dire-rapids
   python -m pip install -e ".[dev,keops]"

Quick Start
-----------

.. code-block:: python

   from dire_rapids import create_dire
   import numpy as np
   
   # Create sample data
   X = np.random.randn(10000, 100)
   
   # Create reducer with automatic implementation and k-NN engine selection
   reducer = create_dire(n_neighbors=32)

   # Or force the k-NN engine independently
   reducer = create_dire(backend='pytorch_cpu', knn_backend='pytorch')

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

Automatic Backend and k-NN Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dire_rapids import create_dire

   # Automatic reducer selection based on hardware
   # Implementation priority: cuVS > PyTorchMemoryEfficient > PyTorch > CPU
   # When cuVS is not available, automatically uses memory-efficient backend
   reducer = create_dire(
       n_neighbors=32,
       memory_efficient=True  # Use memory-efficient variant if needed
   )
   embedding = reducer.fit_transform(X)

``backend`` selects the DiRe implementation. ``knn_backend`` selects the
k-nearest-neighbor engine used inside that implementation. Keep
``knn_backend='auto'`` for the default heuristics, or force ``'pytorch'``,
``'pykeops'``, or ``'cuvs'``. Explicit k-NN backend requests raise if the
requested engine is unavailable or unsupported for the current data.

.. code-block:: python

   # CPU implementation with forced PyTorch k-NN
   reducer = create_dire(backend='pytorch_cpu', knn_backend='pytorch')

   # Optional engines, strict if unavailable
   reducer = create_dire(knn_backend='pykeops')
   reducer = create_dire(knn_backend='cuvs')

**Backend Selection Priority:**

1. RAPIDS cuVS (if available and GPU present)
2. PyTorch Memory-Efficient (if GPU present but cuVS unavailable, or ``memory_efficient=True``)
3. PyTorch Standard (if GPU present and ``memory_efficient=False``)
4. PyTorch CPU (fallback)

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
   print(results['topology']['protocol'])

Topology protocol parameters are exposed as ``topology_n_steps``,
``topology_k_neighbors``, ``topology_density_threshold``,
``topology_overlap_factor``, and ``topology_metrics_only``.

**Metrics:**

* **Distortion**: stress, neighborhood preservation
* **Context**: SVM/kNN classification accuracy
* **Topology**: DTW distances between Betti curves (β₀, β₁) via ripser when available, otherwise a kNN-atlas fallback with union-find and GF(2) bitset elimination

See :doc:`api/dire_rapids.metrics` for full API reference.

Custom Distance Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Custom metrics for k-nearest neighbor computation:

.. code-block:: python

   # L1 distance on the PyTorch k-NN path
   reducer = DiRePyTorch(metric='(x - y).abs().sum(-1)', n_neighbors=32, knn_backend='pytorch')
   embedding = reducer.fit_transform(X)

   # Cosine distance
   def cosine_distance(x, y):
       return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)

   reducer = DiRePyTorch(metric=cosine_distance, knn_backend='pytorch')
   embedding = reducer.fit_transform(X)

**Metric types:** ``None``/``'euclidean'``/``'l2'`` (default), string expressions, callable functions

Note: Layout forces use Euclidean distance regardless of k-NN metric. Custom
metric expressions and callables run on the PyTorch/PyKeOps k-NN paths. cuVS
supports named native metrics only; forced ``knn_backend='cuvs'`` raises for
custom expressions/callables.

ReducerRunner Framework
~~~~~~~~~~~~~~~~~~~~~~~~

Framework for running sklearn-compatible reducers with automatic data loading and metrics.

.. code-block:: python

   from dire_rapids.utils import ReducerRunner, ReducerConfig
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
