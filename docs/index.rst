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

For GPU acceleration with RAPIDS 26.06, use a clean virtual environment and
choose exactly one CUDA-specific extra. The legacy ``rapids`` extra remains a
CUDA 12 alias for backward compatibility.
The core package supports Python 3.10+, while RAPIDS 26.06 requires Python
3.11--3.14.
The CUDA-specific extras are currently unreleased. Install this version from
a clone before choosing one of the CUDA-family commands below:

.. code-block:: bash

   git clone https://github.com/sashakolpakov/dire-rapids.git
   cd dire-rapids

CUDA 13 (recommended for RAPIDS 26.06 with Python 3.14):

.. code-block:: bash

   python -m pip install torch==2.11.0 \
     --index-url https://download.pytorch.org/whl/cu130
   python -m pip install \
     --extra-index-url https://pypi.nvidia.com \
     -e ".[rapids-cu13,keops]"

CUDA 12:

.. code-block:: bash

   python -m pip install torch==2.11.0 \
     --index-url https://download.pytorch.org/whl/cu128
   python -m pip install \
     --extra-index-url https://pypi.nvidia.com \
     -e ".[rapids-cu12,keops]"

Do not combine the CUDA 12 and CUDA 13 extras. Install the pinned PyTorch wheel
first from the index matching the selected CUDA family.

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

   numpy2_rapids
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

RAPIDS 26.06 All-Neighbors Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DiReCuVS`` can use the cuVS all-neighbors API to build a full approximate
k-NN graph (one row per input) without first copying the entire dataset into a
single-GPU ANN index. The partitioned path supports host-backed/out-of-core
data and multiple GPUs. Select ``all_neighbors_algo="brute_force"`` when an
exact local graph is required.

.. code-block:: python

   from dire_rapids import DiReCuVS

   # Automatic preserves the released index-and-search policy.
   reducer = DiReCuVS(cuvs_knn_method="auto")

   # Explicit experimental partitioned/out-of-core construction.
   reducer = DiReCuVS(
       cuvs_knn_method="all_neighbors",
       all_neighbors_algo="nn_descent",
       all_neighbors_n_clusters=16,
       all_neighbors_device_ids=[0, 1],
   )

``cuvs_knn_method`` defaults to ``"auto"``. The remaining defaults are
``all_neighbors_algo="nn_descent"``,
``all_neighbors_n_clusters=1``, ``all_neighbors_device_ids=None``, and
``all_neighbors_algo_params=None``. ``all_neighbors_overlap_factor=None``
selects 0 for one cluster and ``min(2, n_clusters - 1)`` otherwise.
``cuvs_knn_method="auto"`` and ``"index_search"`` both retain the established
index-and-search policy. All-neighbors is explicit opt-in until it clears
frozen neighbor-recall, topology, local, context, and global quality gates;
availability of the API alone does not change existing embeddings.

An H100 A/B audit produced mixed results. At full scale, all-neighbors was
about 26% slower on 10x (0.623 graph overlap) but 1.91x faster on arXiv (0.839
overlap); downstream quality moved in both directions and balanced context
accuracy decreased by 1.62 and 0.84 percentage points, respectively. It was
therefore not promoted to the default, but remains a viable explicit option,
particularly given the arXiv performance. Full observations are recorded in
`PR #12 <https://github.com/sashakolpakov/dire-rapids/pull/12>`_; the harness
and raw-result workflow remain on the separate
`homological-stability-repro test branch
<https://github.com/sashakolpakov/homological-stability-repro/tree/a00aa54949a87ef64e7204ba7434a70155a51c1a>`_.

Partitioning reduces the local graph-builder working set, but the final
``N x k`` index and distance graph must still fit on one GPU.

The legacy automatic index thresholds are:

* fewer than 50,000 rows: exact/flat;
* 50,000 to fewer than 500,000 rows, or more than 500 dimensions: IVF-Flat;
* 500,000 to fewer than 5,000,000 rows at no more than 500 dimensions: IVF-PQ;
* at least 5,000,000 rows with at most 500 dimensions and a supported metric:
  CAGRA;
* otherwise: IVF-PQ.

Fitted Backend Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Requested policy is not a substitute for the algorithm that actually ran.
After fitting, ``DiReCuVS`` exposes ``effective_cuvs_knn_method_`` and
``effective_cuvs_index_type_``. All reducers expose per-stage
``stage_timings_``, ``effective_knn_backend_``, and
``force_chunked_fallback_calls_``. ``get_diagnostics()`` returns these values
as a JSON-serializable dictionary:

.. code-block:: python

   embedding = reducer.fit_transform(X)
   record = reducer.get_diagnostics()
   print(record["cuvs"]["effective_index_type"])
   print(record["stage_timings_seconds"])
   print(record["force_chunked_fallback_calls"])

Forcing an index or opting into all-neighbors can materially change both
runtime and approximation behavior. Benchmark records should retain requested
and effective policies together with recall and downstream embedding-quality
measurements.

See the `cuVS all-neighbors API documentation
<https://docs.rapids.ai/api/cuvs/stable/python_api/neighbors_all_neighbors/>`_
for the underlying RAPIDS interface.

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
* **Topology**: DTW distances between Betti curves (β₀, β₁) via the default kNN-Atlas engine with union-find and GF(2) bitset elimination; Ripser is an explicit reference option

``compute_betti_curve`` tries the GPU Atlas path first when GPU use is enabled,
then the CPU Atlas path. Pass ``prefer_ripser=True`` to request Ripser first.
The former broad ``TOPOLOGY_TUNED`` preset remains withdrawn. The crossed,
held-out replacements have canonical evaluator-specific names::

   from dire_rapids import ATLAS_TUNED, RIPSER_TUNED, create_dire

   atlas_embedding = create_dire(**ATLAS_TUNED).fit_transform(X)
   ripser_embedding = create_dire(**RIPSER_TUNED).fit_transform(X)

The presets are genuinely distinct. Both use ``spread=0.8``, while
``ATLAS_TUNED`` uses ``max_iter_layout=96`` and ``RIPSER_TUNED`` uses 128.
Across six untouched datasets and 20 paired seeds each improved 11/12 cells
against current default DiRe under its respective evaluator, with geometric
discrepancy ratios of 0.908 for both. Against the strongest retained UMAP/t-SNE
method in each cell they won 6/12 cells, with aggregate ratios of 0.884 for
repeated Atlas and 0.891 for the canonical seed-42 Ripser screen. Neither name
promises a universal topology improvement. The initially Atlas-selected
``spread=1.2`` candidate failed to transfer; a subsequent seven-candidate
Atlas refinement selected and held-out-confirmed the 96-iteration layout.

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
