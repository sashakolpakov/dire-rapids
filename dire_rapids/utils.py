# utils.py

"""
Utility classes and functions for dire-rapids package.

This module provides:
- ReducerConfig: Configuration dataclass for dimensionality reduction algorithms
- ReducerRunner: General-purpose runner for dimensionality reduction benchmarking
- Dataset loading utilities for sklearn, cytof, DiRe geometric datasets, and more
"""

import inspect
import os
import re
import time
import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sklearn import datasets as skds

try:
    from scipy import sparse as sp
except ImportError:
    sp = None  # sklearn normally pulls scipy in; keep soft guard


def _identity_transform(X, y):
    return X, y


# --------- minimal display helpers (so .visualize renders in Colab) ---------

def _safe_init_plotly_renderer():
    try:
        import plotly.io as pio  # pylint: disable=import-outside-toplevel
        if pio.renderers.default in (None, "auto"):
            try:
                import google.colab  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
                pio.renderers.default = "colab"
            except ImportError:
                pio.renderers.default = "notebook_connected"
    except ImportError:
        pass


# --------- shared embedding figure builder (scatter + density) ---------
#
# For small/medium embeddings we draw one WebGL marker per point. For large 2D
# embeddings that is both slow (the browser receives every point) and illegible
# (overplotting collapses structure into a solid blob), so we switch to a binned
# density: ``np.histogram2d`` reduces the points to a fixed grid server-side in
# O(n_points), and only that grid (<= n_bins**2 cells, times the number of
# categories) is shipped to the browser. The figure payload is therefore bounded
# regardless of whether there are 50k or 50M points.

# Bins per axis for density rendering; caps the grid (and thus the payload).
# 200 keeps structure crisp while keeping the shipped grid small; per-category
# overlays multiply the grid by the number of categories, so we stay modest.
_DENSITY_BINS = 200
# Above this many categories a per-category overlay is unreadable, so we fall
# back to a single count heatmap.
_MAX_DENSITY_CATEGORIES = 12
# Qualitative palette for per-category density layers (Plotly/D3 style).
_CATEGORY_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
)


def _resolve_use_density(mode, n_dims, n_points, density_threshold):
    """Decide whether to render a binned density rather than a scatter.

    Density is only meaningful in 2D; 3D always falls back to a (subsampled)
    scatter. In ``'auto'`` mode density kicks in once a 2D embedding exceeds
    ``density_threshold`` points.
    """
    if mode not in ("auto", "scatter", "density"):
        raise ValueError(f"mode must be 'auto', 'scatter' or 'density', got {mode!r}")
    if mode == "scatter" or n_dims != 2:
        return False
    if mode == "density":
        return True
    return n_points > density_threshold


def _shared_bin_edges(x, y, n_bins):
    """Common bin edges so every per-category histogram aligns on one grid."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if x_max <= x_min:  # guard degenerate (zero-width) ranges
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    return (np.linspace(x_min, x_max, n_bins + 1),
            np.linspace(y_min, y_max, n_bins + 1))


def _density_traces_2d(embedding, labels, categorical_labels, n_bins):
    """Build bounded-payload density traces for a 2D embedding.

    Returns a single count/mean heatmap when there are no labels, continuous
    labels, or too many categories; otherwise one filled-contour layer per
    category (the per-category density overlay).
    """
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    x = np.asarray(embedding[:, 0], dtype=float)
    y = np.asarray(embedding[:, 1], dtype=float)
    x_edges, y_edges = _shared_bin_edges(x, y, n_bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    def _heatmap(z, colorbar_title):
        return go.Heatmap(
            x=x_centers, y=y_centers, z=z,
            colorscale="Viridis", colorbar={"title": colorbar_title},
            hoverongaps=False,
        )

    # No labels -> single count-density heatmap. Counts are integers; cast so
    # they serialize compactly when the figure is shipped to the browser.
    if labels is None:
        counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        return [_heatmap(counts.T.astype(np.int32), "Count")]

    labels = np.asarray(labels)

    # Continuous labels -> mean-label-per-bin heatmap.
    if not categorical_labels:
        counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        sums, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges),
                                    weights=labels.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / counts, np.nan)
        return [_heatmap(mean.T, "Mean label")]

    # Categorical labels: one density layer per category (overlay), unless there
    # are too many categories to read.
    unique = np.unique(labels)
    if len(unique) > _MAX_DENSITY_CATEGORIES:
        counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        return [_heatmap(counts.T.astype(np.int32), "Count")]

    traces = []
    for idx, label in enumerate(unique):
        mask = labels == label
        counts, _, _ = np.histogram2d(x[mask], y[mask], bins=(x_edges, y_edges))
        peak = float(counts.max())
        if peak <= 0:
            continue
        color = _CATEGORY_COLORS[idx % len(_CATEGORY_COLORS)]
        traces.append(go.Contour(
            x=x_centers, y=y_centers, z=counts.T.astype(np.int32),
            name=str(label), showscale=False, showlegend=True, opacity=0.55,
            # transparent -> category color, so empty bins stay invisible
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, color]],
            contours={"coloring": "fill", "start": peak * 0.2,
                      "end": peak, "size": max(peak / 5.0, 1.0)},
            line={"width": 0},
            hovertemplate=f"{label}<extra></extra>",
        ))
    return traces


def _scatter_traces(embedding, labels, categorical_labels, n_dims, point_size):
    """Build WebGL scatter traces (2D Scattergl / 3D Scatter3d)."""
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    scatter = go.Scattergl if n_dims == 2 else go.Scatter3d

    def coords(arr):
        xyz = {"x": arr[:, 0], "y": arr[:, 1]}
        if n_dims == 3:
            xyz["z"] = arr[:, 2]
        return xyz

    if labels is None:
        return [scatter(**coords(embedding), mode="markers",
                        marker={"size": point_size, "opacity": 0.7})]

    labels = np.asarray(labels)
    if not categorical_labels:
        return [scatter(**coords(embedding), mode="markers",
                        marker={"size": point_size, "color": labels,
                                "colorscale": "Viridis",
                                "colorbar": {"title": "Label Value"},
                                "showscale": True, "opacity": 0.8})]

    unique = np.unique(labels)
    if len(unique) > 20:
        label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
        colors = np.array([label_to_idx[lbl] for lbl in labels])
        return [scatter(**coords(embedding), mode="markers",
                        marker={"size": point_size, "color": colors,
                                "colorscale": "Viridis", "showscale": True,
                                "opacity": 0.8},
                        text=[f"Label: {lbl}" for lbl in labels],
                        hovertemplate="%{text}<extra></extra>")]

    traces = []
    for label in unique:
        mask = labels == label
        traces.append(scatter(**coords(embedding[mask]), mode="markers",
                              name=str(label),
                              marker={"size": point_size, "opacity": 0.8}))
    return traces


def build_embedding_figure(
    embedding,
    labels=None,
    *,
    title="Embedding",
    n_dims=None,
    categorical_labels=True,
    mode="auto",
    density_threshold=50000,
    max_points=10000,
    n_bins=_DENSITY_BINS,
    point_size=None,
    width=None,
    height=None,
    seed=42,
    logger=None,
):
    """Build a Plotly figure for a 2D/3D embedding (scatter or binned density).

    Parameters
    ----------
    embedding : ndarray of shape (n_points, 2 or 3)
        The low-dimensional layout to plot.
    labels : array-like of shape (n_points,), optional
        Per-point labels used for coloring (scatter) or density layers.
    title : str, default="Embedding"
        Base title; the render type and point count are appended.
    n_dims : int, optional
        Embedding dimensionality; inferred from ``embedding`` if None.
    categorical_labels : bool, default=True
        Treat labels as discrete classes (per-category colors / density layers)
        rather than a continuous scalar (single colorbar / mean heatmap).
    mode : {'auto', 'scatter', 'density'}, default='auto'
        ``'auto'`` switches to density once a 2D embedding exceeds
        ``density_threshold`` points; ``'density'`` forces density (2D only,
        falls back to scatter in 3D); ``'scatter'`` always draws markers.
    density_threshold : int, default=50000
        Point count above which ``'auto'`` mode uses density.
    max_points : int, default=10000
        Subsample cap for scatter rendering (density uses all points).
    n_bins : int, default=300
        Bins per axis for density; bounds the figure payload.
    point_size : int, optional
        Marker size; defaults to 4 (2D) / 2 (3D) when None.
    width, height : int, optional
        Figure size overrides.
    seed : int, default=42
        RNG seed for reproducible scatter subsampling.
    logger : logging.Logger, optional
        Used for warnings; falls back to silent when None.

    Returns
    -------
    plotly.graph_objects.Figure or None
        ``None`` if the embedding is not 2D/3D.
    """
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    def _warn(msg):
        if logger is not None:
            logger.warning(msg)

    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[1] not in (2, 3):
        _warn(f"Cannot visualize embedding with shape {embedding.shape}")
        return None
    n_points = embedding.shape[0]
    if n_dims is None:
        n_dims = embedding.shape[1]

    if mode == "density" and n_dims != 2:
        _warn("density mode is only supported for 2D embeddings; using scatter")

    if _resolve_use_density(mode, n_dims, n_points, density_threshold):
        traces = _density_traces_2d(embedding, labels, categorical_labels, n_bins)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{title} - 2D Density ({n_points:,} points, {n_bins}×{n_bins} bins)",
            xaxis_title="Dimension 1", yaxis_title="Dimension 2",
            width=width or 800, height=height or 600, hovermode="closest",
        )
        return fig

    # Scatter (subsample if larger than max_points).
    if n_points > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_points, max_points, replace=False)
        emb_vis = embedding[idx]
        labels_vis = np.asarray(labels)[idx] if labels is not None else None
    else:
        emb_vis = embedding
        labels_vis = labels

    if point_size is None:
        point_size = 4 if n_dims == 2 else 2
    traces = _scatter_traces(emb_vis, labels_vis, categorical_labels, n_dims, point_size)
    fig = go.Figure(data=traces)
    if n_dims == 2:
        fig.update_layout(
            title=f"{title} - 2D Embedding",
            xaxis_title="Dimension 1", yaxis_title="Dimension 2",
            width=width or 800, height=height or 600, hovermode="closest",
        )
    else:
        fig.update_layout(
            title=f"{title} - 3D Embedding",
            scene={"xaxis_title": "Dimension 1", "yaxis_title": "Dimension 2",
                   "zaxis_title": "Dimension 3"},
            width=width or 900, height=height or 700,
        )
    return fig


def _infer_categorical(labels):
    """Heuristic: strings/objects/bools are categorical; numeric is continuous.

    Matches the prior ``px.scatter`` behavior where integer class labels were
    rendered with a continuous colorbar.
    """
    if labels is None:
        return True
    return np.asarray(labels).dtype.kind in ("U", "S", "O", "b")


def _display_obj(obj):  # pylint: disable=too-many-return-statements
    """Display an object using appropriate renderer (plotly, matplotlib, IPython)."""
    if obj is None:
        return False
    if isinstance(obj, (list, tuple)):
        shown = False
        for it in obj:
            shown = _display_obj(it) or shown
        return shown
    # Plotly
    try:
        import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel
        if isinstance(obj, go.Figure):
            _safe_init_plotly_renderer()
            obj.show()
            return True
    except (ImportError, AttributeError):
        pass
    # Matplotlib
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib.figure import Figure  # pylint: disable=import-outside-toplevel
        from matplotlib.axes import Axes  # pylint: disable=import-outside-toplevel
        if isinstance(obj, (Figure, Axes)):
            plt.show()
            return True
    except (ImportError, AttributeError):
        pass
    # HTML / str
    if isinstance(obj, (str, bytes)):
        s = obj.decode("utf-8", "ignore") if isinstance(obj, bytes) else obj
        if "<" in s and ">" in s:
            try:
                from IPython.display import display, HTML  # pylint: disable=import-outside-toplevel
                display(HTML(s))
            except ImportError:
                print(s)  # Fallback to print if IPython not available
        else:
            print(s)
        return True
    try:
        try:
            from IPython.display import display  # pylint: disable=import-outside-toplevel
            display(obj)
        except ImportError:
            print(obj)  # Fallback to print if IPython not available
        return True
    except (ImportError, AttributeError, TypeError):
        return False


# --------- sklearn resolution ---------

_SKLEARN_ALIASES = {
    # loaders
    "iris": "load_iris",
    "digits": "load_digits",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "diabetes": "load_diabetes",
    "linnerud": "load_linnerud",
    # generators
    "blobs": "make_blobs",
    "classification": "make_classification",
    "multilabel_classification": "make_multilabel_classification",
    "moons": "make_moons",
    "circles": "make_circles",
    "s_curve": "make_s_curve",
    "swiss_roll": "make_swiss_roll",
    "gaussian_quantiles": "make_gaussian_quantiles",
    "low_rank_matrix": "make_low_rank_matrix",
    "spd_matrix": "make_spd_matrix",
    "sparse_spd_matrix": "make_sparse_spd_matrix",
}

def _normalize_key(s):
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower())

def _resolve_sklearn_function(name):
    n = _normalize_key(name)
    if n.startswith(("load_", "fetch_", "make_")):
        fn = getattr(skds, n, None)
        if callable(fn):
            return n, fn
    alias = _SKLEARN_ALIASES.get(n)
    if alias and callable(getattr(skds, alias, None)):
        return alias, getattr(skds, alias)
    for pref in ("load_", "fetch_", "make_"):
        cand = pref + n
        fn = getattr(skds, cand, None)
        if callable(fn):
            return cand, fn
    candidates = [
        (attr, getattr(skds, attr))
        for attr in dir(skds)
        if attr.lower().endswith(n) and callable(getattr(skds, attr))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(a for a, _ in candidates[:6])
        raise ValueError(f"Ambiguous sklearn dataset '{name}'. Candidates: {names} ...")
    all_names = ", ".join(a for a in dir(skds) if a.startswith(("load_", "fetch_", "make_")))
    raise ValueError(f"Unknown sklearn dataset '{name}'. Available include: {all_names}")


def _to_Xy_from_obj(obj):
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        X = obj[0]
        y = obj[1] if len(obj) > 1 else None
        return _coerce_Xy(X, y)
    if hasattr(obj, "get"):
        data = obj.get("data", None)
        target = obj.get("target", None)
        images = obj.get("images", None)
        if data is None and images is not None:
            imgs = np.asarray(images)
            data = imgs.reshape(len(imgs), -1)
        return _coerce_Xy(data, target)
    if hasattr(obj, "shape"):
        return _coerce_Xy(obj, None)
    raise ValueError("Unsupported sklearn return type; cannot coerce to (X, y).")


def _coerce_Xy(X, y):
    if isinstance(X, list) and X and isinstance(X[0], str):
        raise TypeError("Loaded dataset contains text data; vectorize first.")
    if sp is not None and sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        uniq = {v: i for i, v in enumerate(np.unique(y))}
        y = np.array([uniq[v] for v in y], dtype=np.int32)
    return X, y


def _load_sklearn_any(name, **kwargs):
    _, fn = _resolve_sklearn_function(name)
    try:
        sig = inspect.signature(fn)
        if "return_X_y" in sig.parameters:
            obj = fn(return_X_y=True, **kwargs)
            X, y = _to_Xy_from_obj(obj)
        else:
            obj = fn(**kwargs)
            X, y = _to_Xy_from_obj(obj)
    except TypeError:
        obj = fn()
        X, y = _to_Xy_from_obj(obj)
    return X, y


# --------- file loader ---------

def _load_file(path, **kwargs):
    path = str(path)
    ext = Path(path).suffix.lower()

    # pandas is an optional dep (ships in [viz]). Import lazily so users
    # who never touch tabular file formats don't need it installed.
    import pandas as pd  # pylint: disable=import-outside-toplevel

    if ext == ".csv":
        df = pd.read_csv(path)
        label_col = kwargs.pop("label_column", None)
        if label_col and label_col in df.columns:
            y = df[label_col].to_numpy()
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        else:
            y = None
            X = df.to_numpy(dtype=np.float32)
        return X, y

    if ext == ".parquet":
        df = pd.read_parquet(path)
        label_col = kwargs.pop("label_column", None)
        if label_col and label_col in df.columns:
            y = df[label_col].to_numpy()
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        else:
            y = None
            X = df.to_numpy(dtype=np.float32)
        return X, y

    if ext == ".npy":
        X = np.load(path, mmap_mode="r")
        y = None
        labels_path = kwargs.pop("labels_path", None)
        if labels_path:
            y = np.load(labels_path, mmap_mode="r")
        return np.asarray(X, dtype=np.float32), y

    if ext == ".npz":
        f = np.load(path, mmap_mode="r")
        if "X" not in f:
            raise ValueError(".npz must contain key 'X' (and optionally 'y').")
        X = np.asarray(f["X"], dtype=np.float32)
        y = f["y"] if "y" in f else None
        return X, y

    raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .npy, .npz, or .parquet.")


# --------- DiRe geometric datasets ---------

def rand_point_disk(n_features, n_samples=1, rng=None):
    """Generate uniformly distributed points in n-dimensional unit disk."""
    if rng is None:
        rng = np.random.default_rng()
    prepts = rng.standard_normal((n_samples, n_features))
    prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
    rads = np.sqrt(rng.random(n_samples)).reshape(-1, 1)
    pts = prepts * rads / prenorms
    return pts


def rand_point_sphere(n_features, n_samples=1, rng=None):
    """Generate uniformly distributed points on n-dimensional unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    prepts = rng.standard_normal((n_samples, n_features))
    prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
    pts = prepts / prenorms
    return pts


class elgen:
    """Ellipsoid generator - transforms sphere points to ellipsoid."""
    def __init__(self, a):
        a = np.array(a)
        themat = np.diag(1 / (a * a))
        L = np.linalg.inv(np.linalg.cholesky(themat).T)
        self.L = L

    def __call__(self, ar):
        return (self.L @ ar.T).T


def rand_point_ell(semi_axes, n_features, n_samples=1, rng=None):
    """Generate uniformly distributed points on n-dimensional ellipsoid with semi-axes."""
    spts = rand_point_sphere(n_features, n_samples, rng=rng)
    eg = elgen(semi_axes)
    return eg(spts)


def _load_dire_dataset(name, **kwargs):
    """
    Load DiRe geometric datasets.

    Supported:
    - 'disk_uniform': Uniform in n-dimensional unit disk
    - 'sphere_uniform': Uniform on n-dimensional unit sphere
    - 'ellipsoid_uniform': Uniform on n-dimensional ellipsoid

    Options:
    - n_samples (default 1000)
    - n_features (default 10)
    - semi_axes (for ellipsoid, default [1, 2, ..., n])
    - random_state
    """
    key = _normalize_key(name)

    n_samples = kwargs.pop('n_samples', 1000)
    n_features = kwargs.pop('n_features', 10)
    random_state = kwargs.pop('random_state', None)

    rng = np.random.default_rng(random_state)

    if key == 'disk_uniform':
        X = rand_point_disk(n_features, n_samples, rng=rng)
    elif key == 'sphere_uniform':
        X = rand_point_sphere(n_features, n_samples, rng=rng)
    elif key == 'ellipsoid_uniform':
        semi_axes = kwargs.pop('semi_axes', None)
        if semi_axes is not None:
            n_features = len(semi_axes)  # Infer n_features from semi_axes
        else:
            semi_axes = list(range(1, n_features + 1))  # Default semi_axes
        X = rand_point_ell(semi_axes, n_features, n_samples, rng=rng)
    else:
        raise ValueError(
            f"Unknown DiRe dataset '{name}'. Options: 'disk_uniform', 'sphere_uniform', 'ellipsoid_uniform'"
        )

    return X.astype(np.float32), None


# --------- cytof scheme (Levine13/32) ---------

_DEF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "reducer_runner", "cytof")
os.makedirs(_DEF_CACHE, exist_ok=True)

def _download(url, dest, *, overwrite=False):
    if (not overwrite) and os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest

def _safe_gunzip(path):
    if path.endswith(".gz"):
        out = path[:-3]
        if not os.path.exists(out):
            with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out
    return path

_CYTOF_REGISTRY = {
    "levine13": {
        "urls": [
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-13-dim/master/data/Levine_13dim.fcs",
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-13-dim/master/data/Levine_13dim.txt",
        ],
        "label_column": "label",
        "drop_columns": ("label", "individual"),
    },
    "levine32": {
        "urls": [
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-32-dim/master/data/Levine_32dim.fcs",
        ],
        "label_column": "label",
        "drop_columns": ("label", "individual"),
    },
}

def _load_cytof(name, **kwargs):
    """
    CyTOF loader:
      - 'levine13'
      - 'levine32'
    via built-in URLs/caching
    Supports .txt/.tsv/.csv (pandas).
    Options:
      - url / file / cache_dir
      - label_column (for txt/csv/tsv)
      - drop_columns
      - arcsinh_cofactor (if raw)
    """

    # pandas ships in the [viz] extra; import lazily so users who never load
    # CyTOF datasets don't need it installed.
    import pandas as pd  # pylint: disable=import-outside-toplevel

    key = _normalize_key(name)
    spec = _CYTOF_REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"Unknown cytof dataset '{name}'. Options: {tuple(_CYTOF_REGISTRY.keys())}")

    cache_dir = kwargs.pop("cache_dir", _DEF_CACHE)
    url = kwargs.pop("url", None)
    label_col = kwargs.pop("label_column", spec.get("label_column", "label"))
    drop_cols = tuple(kwargs.pop("drop_columns", spec.get("drop_columns", (label_col,))))
    drop_unassigned = bool(kwargs.pop("drop_unassigned", False))
    arcsinh_cofactor = kwargs.pop("arcsinh_cofactor", None)
    local_path = kwargs.pop("file", None)

    # Resolve local or download
    if local_path is None:
        urls = [url] if url else spec.get("urls", [])
        if not urls:
            raise ValueError(f"cytof:{name} requires 'url' or local 'file' path.")
        last_err = None
        for u in urls:
            try:
                fname = os.path.join(cache_dir, os.path.basename(u.split("?")[0]))
                local_path = _download(u, fname)
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_err = e
                local_path = None
        if local_path is None:
            raise RuntimeError(f"Failed to download cytof:{name}: {last_err}") from last_err

    path = _safe_gunzip(local_path)
    ext = Path(path).suffix.lower()

    # ---------- FCS via flowio ----------
    if ext == ".fcs":
        try:
            import flowio  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("flowio required for FCS files. Install with: pip install flowio") from exc

        fcs = flowio.FlowData(path)
        data = fcs.as_array()  # Get 2D numpy array with preprocessing

        # Get channel names from pnn_labels (parameter names)
        channel_names = fcs.pnn_labels if fcs.pnn_labels else [f'Ch{i}' for i in range(fcs.channel_count)]

        # Create DataFrame from FCS data
        df = pd.DataFrame(data, columns=channel_names)

        # Drop rows with null labels if requested
        if drop_unassigned and label_col in df.columns:
            before = len(df)
            df = df[df[label_col].notna()].copy()
            after = len(df)
            print(f"[cytof] dropped {before - after} rows with null labels")

        y = df[label_col].to_numpy() if label_col in df.columns else None
        drop = [c for c in drop_cols if c in df.columns]
        Xdf = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        X = Xdf.to_numpy(dtype=np.float32, copy=False)

        if (arcsinh_cofactor is not None) and arcsinh_cofactor > 0:
            X = np.arcsinh(X / float(arcsinh_cofactor)).astype(np.float32)

        # map string labels to ints
        if y is not None:
            y = np.asarray(y)
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = {v: i for i, v in enumerate(np.unique(y))}
                y = np.array([uniq[v] for v in y], dtype=np.int32)
            elif y.dtype.kind == "f":  # floating point labels
                y = y.astype(np.int32)

        return X, y

    # ---------- TXT/TSV/CSV via pandas ----------
    if ext in (".txt", ".tsv", ".csv"):
        sep = "\t" if ext in (".txt", ".tsv") else ","
        df = pd.read_csv(path, sep=sep)

        # Drop rows with null labels if requested
        if drop_unassigned and label_col in df.columns:
            before = len(df)
            df = df[df[label_col].notna()].copy()
            after = len(df)
            print(f"[cytof] dropped {before - after} rows with null labels")

        y = df[label_col].to_numpy() if label_col in df.columns else None
        drop = [c for c in drop_cols if c in df.columns]
        Xdf = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        X = Xdf.to_numpy(dtype=np.float32, copy=False)

        if (arcsinh_cofactor is not None) and arcsinh_cofactor > 0:
            X = np.arcsinh(X / float(arcsinh_cofactor)).astype(np.float32)

        # map string labels to ints
        if y is not None:
            y = np.asarray(y)
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = {v: i for i, v in enumerate(np.unique(y))}
                y = np.array([uniq[v] for v in y], dtype=np.int32)
            elif y.dtype.kind == "f":  # floating point labels
                y = y.astype(np.int32)

        return X, y

    raise ValueError(f"Unsupported cytof file: {path} (use .fcs, .txt/.tsv, or .csv)")



# --------- ReducerConfig ---------

@dataclass
class ReducerConfig:
    """
    Configuration for a dimensionality reduction algorithm.

    All fields are mutable and can be changed after creation:
        config.visualize = True
        config.categorical_labels = False
        config.max_points = 20000
    """
    name: str
    reducer_class: type
    reducer_kwargs: dict
    visualize: bool = False
    categorical_labels: bool = True  # False for regression-style labels (swiss_roll, etc.)
    max_points: int = 10000  # Max points for scatter visualization (subsamples if larger)
    mode: str = "auto"  # 'auto' | 'scatter' | 'density' rendering for visualization
    density_threshold: int = 50000  # 'auto' switches 2D to density above this many points


# --------- selector parsing ---------

def _parse_selector(selector):
    s = selector.strip()
    p = Path(s)
    if p.exists() or re.search(r"\.(csv|np[yz]|parquet)$", s, re.I):
        return "file", s
    m = re.match(r"^(?P<scheme>[A-Za-z0-9_]+)[:\.](?P<name>.+)$", s)
    if m:
        return m.group("scheme").lower(), m.group("name").strip()
    return "sklearn", s


# --------- Runner ---------

@dataclass
class ReducerRunner:
    """
    General-purpose runner for dimensionality reduction algorithms.

    Supports:
    - DiRe (create_dire, DiRePyTorch, DiRePyTorchMemoryEfficient, DiReCuVS)
    - cuML (UMAP, TSNE)
    - scikit-learn (any TransformerMixin-compatible class)

    Parameters
    ----------
    config : ReducerConfig
        Configuration object containing reducer_class, reducer_kwargs, name, and visualize flag.
    """
    config: ReducerConfig

    def __post_init__(self):
        """Validate that config is provided."""
        if self.config is None:
            raise ValueError("Must provide 'config' (ReducerConfig)")

    def _get_reducer_info(self):
        """Extract reducer info from config."""
        return (
            self.config.name,
            self.config.reducer_class,
            self.config.reducer_kwargs,
            self.config.visualize,
            self.config.categorical_labels,
            self.config.max_points,
            self.config.mode,
            self.config.density_threshold,
        )

    def run(self, dataset, *, dataset_kwargs=None, transform=None):
        """
        Run dimensionality reduction on specified dataset.

        Parameters
        ----------
        dataset : str
            Dataset selector (sklearn:name, openml:name, cytof:name, dire:name, file:path)
        dataset_kwargs : dict, optional
            Arguments for dataset loader
        transform : callable, optional
            Custom transform function (X, y) -> (X', y')

        Returns
        -------
        dict
            Results containing:
            - embedding: reduced data
            - labels: data labels
            - reducer: fitted reducer instance
            - fit_time_sec: time taken for fit_transform
            - dataset_info: dataset metadata
        """
        # Get reducer configuration
        (reducer_name, reducer_class, reducer_kwargs, should_visualize,
         categorical_labels, max_points, mode, density_threshold) = self._get_reducer_info()

        scheme, name = _parse_selector(dataset)
        dataset_kwargs = dataset_kwargs or {}

        if scheme == "sklearn":
            X, y = _load_sklearn_any(name, **dataset_kwargs)
        elif scheme == "file":
            X, y = _load_file(name, **dataset_kwargs)
        elif scheme == "openml":
            from sklearn.datasets import fetch_openml  # pylint: disable=import-outside-toplevel
            try:
                data_id = int(str(name))
                ds = fetch_openml(data_id=data_id, return_X_y=True, **dataset_kwargs)
            except (ValueError, TypeError):
                ds = fetch_openml(name=name, return_X_y=True, **dataset_kwargs)
            X, y = _coerce_Xy(ds[0], ds[1])
        elif scheme == "cytof":
            X, y = _load_cytof(name, **dataset_kwargs)
        elif scheme == "dire":
            X, y = _load_dire_dataset(name, **dataset_kwargs)
        else:
            raise ValueError(f"Unsupported scheme '{scheme}'. Use 'sklearn', 'openml', 'cytof', 'dire', 'file'.")

        T = transform or _identity_transform
        X, y = T(X, y)

        # Instantiate reducer (handles both classes and factory functions)
        if callable(reducer_class):
            reducer = reducer_class(**reducer_kwargs)
        else:
            raise TypeError(f"reducer_class must be callable, got {type(reducer_class)}")

        t0 = time.perf_counter()
        embedding = reducer.fit_transform(X)
        t1 = time.perf_counter()

        # Handle visualization
        if should_visualize:
            # Only use ReducerRunner's plotly visualization (not the reducer's built-in visualize)
            n_dims = embedding.shape[1] if len(embedding.shape) > 1 else 1
            if n_dims in (2, 3):
                try:
                    self._visualize_with_plotly(embedding, y, reducer_name, n_dims,
                                                categorical_labels, max_points,
                                                mode, density_threshold)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"[WARNING] plotly visualization failed: {e}")

        return {
            "embedding": embedding,
            "labels": y,
            "reducer": reducer,
            "fit_time_sec": float(t1 - t0),
            "dataset_info": {
                "selector": dataset,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
        }

    def _visualize_with_plotly(self, embedding, labels, title, n_dims,
                               categorical_labels=True, max_points=10000,
                               mode="auto", density_threshold=50000):
        """
        Create and display a plotly visualization for 2D or 3D embeddings.

        Uses WebGL scatter (Scattergl/Scatter3d) for moderate point counts. For
        large 2D embeddings (see ``mode``/``density_threshold``) it switches to a
        binned density so the figure payload stays bounded; see
        :func:`build_embedding_figure`.
        """
        try:
            import plotly.graph_objects  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError:
            print("[WARNING] plotly not installed. Install with: pip install plotly")
            return

        _safe_init_plotly_renderer()

        fig = build_embedding_figure(
            embedding, labels, title=title, n_dims=n_dims,
            categorical_labels=categorical_labels, mode=mode,
            density_threshold=density_threshold, max_points=max_points,
        )
        if fig is not None:
            fig.show()

    @staticmethod
    def available_sklearn():
        """Return available sklearn dataset loaders, fetchers, and generators."""
        loads = tuple(a for a in dir(skds) if a.startswith("load_") and callable(getattr(skds, a)))
        fetches = tuple(a for a in dir(skds) if a.startswith("fetch_") and callable(getattr(skds, a)))
        makes = tuple(a for a in dir(skds) if a.startswith("make_") and callable(getattr(skds, a)))
        return {"load": loads, "fetch": fetches, "make": makes}

    @staticmethod
    def available_cytof():
        """Return available CyTOF datasets."""
        return tuple(_CYTOF_REGISTRY.keys())
