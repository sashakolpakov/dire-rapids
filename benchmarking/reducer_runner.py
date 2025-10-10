# reducer_runner.py
from __future__ import annotations

import inspect
import os
import re
import time
import gzip
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from IPython.display import display, HTML  # for Colab/Notebook visualize

from sklearn import datasets as skds

try:
    from scipy import sparse as sp
except Exception:
    sp = None  # sklearn normally pulls scipy in; keep soft guard

TransformFn = Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]


def _identity_transform(X: np.ndarray, y: Optional[np.ndarray]):
    return X, y


# --------- minimal display helpers (so .visualize renders in Colab) ---------

def _safe_init_plotly_renderer():
    try:
        import plotly.io as pio
        if pio.renderers.default in (None, "auto"):
            try:
                import google.colab  # noqa: F401
                pio.renderers.default = "colab"
            except Exception:
                pio.renderers.default = "notebook_connected"
    except Exception:
        pass

def _display_obj(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (list, tuple)):
        shown = False
        for it in obj:
            shown = _display_obj(it) or shown
        return shown
    # Plotly
    try:
        import plotly.graph_objects as go
        if isinstance(obj, go.Figure):
            _safe_init_plotly_renderer()
            obj.show()
            return True
    except Exception:
        pass
    # Matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        if isinstance(obj, (Figure, Axes)):
            plt.show()
            return True
    except Exception:
        pass
    # HTML / str
    if isinstance(obj, (str, bytes)):
        s = obj.decode("utf-8", "ignore") if isinstance(obj, bytes) else obj
        if "<" in s and ">" in s:
            display(HTML(s))
        else:
            print(s)
        return True
    try:
        display(obj)
        return True
    except Exception:
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

def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower())

def _resolve_sklearn_function(name: str):
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


def _to_Xy_from_obj(obj) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def _coerce_Xy(X, y) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def _load_sklearn_any(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

def _load_file(path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        import pandas as pd
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
        import pandas as pd
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

def rand_point_disk(n_features, n_samples=1):
    """Generate uniformly distributed points in n-dimensional unit disk."""
    prepts = np.random.randn(n_samples, n_features)
    prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
    rads = np.sqrt(np.random.rand(n_samples)).reshape(-1, 1)
    pts = prepts * rads / prenorms
    return pts


def rand_point_sphere(n_features, n_samples=1):
    """Generate uniformly distributed points on n-dimensional unit sphere."""
    prepts = np.random.randn(n_samples, n_features)
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


def rand_point_ell(semi_axes, n_features, n_samples=1):
    """Generate uniformly distributed points on n-dimensional ellipsoid with semi-axes."""
    spts = rand_point_sphere(n_features, n_samples)
    eg = elgen(semi_axes)
    return eg(spts)


def _load_dire_dataset(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    if random_state is not None:
        np.random.seed(random_state)

    if key == 'disk_uniform':
        X = rand_point_disk(n_features, n_samples)
    elif key == 'sphere_uniform':
        X = rand_point_sphere(n_features, n_samples)
    elif key == 'ellipsoid_uniform':
        semi_axes = kwargs.pop('semi_axes', None)
        if semi_axes is not None:
            n_features = len(semi_axes)  # Infer n_features from semi_axes
        else:
            semi_axes = list(range(1, n_features + 1))  # Default semi_axes
        X = rand_point_ell(semi_axes, n_features, n_samples)
    else:
        raise ValueError(
            f"Unknown DiRe dataset '{name}'. Options: 'disk_uniform', 'sphere_uniform', 'ellipsoid_uniform'"
        )

    return X.astype(np.float32), None


# --------- cytof scheme (Levine13/32) ---------

_DEF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "reducer_runner", "cytof")
os.makedirs(_DEF_CACHE, exist_ok=True)

def _download(url: str, dest: str, *, overwrite=False):
    if (not overwrite) and os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest

def _safe_gunzip(path: str) -> str:
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

def _load_cytof(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
    import pandas as pd

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
            except Exception as e:
                last_err = e
                local_path = None
        if local_path is None:
            raise RuntimeError(f"Failed to download cytof:{name}: {last_err}")

    path = _safe_gunzip(local_path)
    ext = Path(path).suffix.lower()

    # ---------- FCS via flowio ----------
    if ext == ".fcs":
        try:
            import flowio
        except ImportError:
            raise ImportError("flowio required for FCS files. Install with: pip install flowio")

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
    """Configuration for a dimensionality reduction algorithm."""
    name: str
    reducer_class: type
    reducer_kwargs: Dict[str, Any]
    visualize: bool = False


# --------- selector parsing ---------

def _parse_selector(selector: str) -> Tuple[str, str]:
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
    config : ReducerConfig, optional
        Configuration object containing reducer_class, reducer_kwargs, name, and visualize flag.
        If provided, individual reducer_class/reducer_kwargs/call_visualize params are ignored.
    reducer_class : type or callable, optional
        Reducer class or factory function (e.g., create_dire, UMAP, TSNE).
        Used only if config is not provided.
    reducer_kwargs : dict, optional
        Keyword arguments to pass to reducer constructor.
        Used only if config is not provided.
    call_visualize : bool
        Whether to call .visualize() method if available (legacy parameter).
        Overridden by config.visualize if config is provided.
    default_transform : callable
        Default transform to apply to data before reduction
    """
    config: Optional[ReducerConfig] = None
    reducer_class: Optional[type] = None
    reducer_kwargs: Dict[str, Any] = field(default_factory=dict)
    call_visualize: bool = True
    default_transform: TransformFn = _identity_transform

    def __post_init__(self):
        """Validate that either config or reducer_class is provided."""
        if self.config is None and self.reducer_class is None:
            raise ValueError("Must provide either 'config' or 'reducer_class'")
        if self.config is not None and self.reducer_class is not None:
            raise ValueError("Cannot provide both 'config' and 'reducer_class'")

    def _get_reducer_info(self) -> Tuple[str, type, Dict[str, Any], bool]:
        """Extract reducer info from config or individual parameters."""
        if self.config is not None:
            return (
                self.config.name,
                self.config.reducer_class,
                self.config.reducer_kwargs,
                self.config.visualize
            )
        else:
            name = getattr(self.reducer_class, '__name__', 'Unknown')
            return (name, self.reducer_class, self.reducer_kwargs, self.call_visualize)

    def run(
        self,
        dataset: str,
        *,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[TransformFn] = None,
    ) -> Dict[str, Any]:
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
        reducer_name, reducer_class, reducer_kwargs, should_visualize = self._get_reducer_info()

        scheme, name = _parse_selector(dataset)
        dataset_kwargs = dataset_kwargs or {}

        if scheme == "sklearn":
            X, y = _load_sklearn_any(name, **dataset_kwargs)
        elif scheme == "file":
            X, y = _load_file(name, **dataset_kwargs)
        elif scheme == "openml":
            from sklearn.datasets import fetch_openml
            try:
                data_id = int(str(name))
                ds = fetch_openml(data_id=data_id, return_X_y=True, **dataset_kwargs)
            except Exception:
                ds = fetch_openml(name=name, return_X_y=True, **dataset_kwargs)
            X, y = _coerce_Xy(ds[0], ds[1])
        elif scheme == "cytof":
            X, y = _load_cytof(name, **dataset_kwargs)
        elif scheme == "dire":
            X, y = _load_dire_dataset(name, **dataset_kwargs)
        else:
            raise ValueError(f"Unsupported scheme '{scheme}'. Use 'sklearn', 'openml', 'cytof', 'dire', 'file'.")

        T = transform or self.default_transform
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
            # First try the reducer's built-in visualize method
            if hasattr(reducer, "visualize"):
                try:
                    vis = reducer.visualize(labels=y)
                    shown = _display_obj(vis)
                    if not shown:
                        try:
                            import matplotlib.pyplot as plt
                            plt.show()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARNING] .visualize failed: {e}")

            # If embedding is 2D or 3D, create plotly visualization
            n_dims = embedding.shape[1] if len(embedding.shape) > 1 else 1
            if n_dims in (2, 3):
                try:
                    self._visualize_with_plotly(embedding, y, reducer_name, n_dims)
                except Exception as e:
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

    def _visualize_with_plotly(
        self,
        embedding: np.ndarray,
        labels: Optional[np.ndarray],
        title: str,
        n_dims: int
    ):
        """Create and display plotly visualization for 2D or 3D embeddings."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[WARNING] plotly not installed. Install with: pip install plotly")
            return

        _safe_init_plotly_renderer()

        if n_dims == 2:
            # 2D scatter plot
            if labels is not None:
                unique_labels = np.unique(labels)
                fig = go.Figure()
                for label in unique_labels:
                    mask = labels == label
                    fig.add_trace(go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode='markers',
                        name=str(label),
                        marker=dict(size=5)
                    ))
            else:
                fig = go.Figure(data=go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(size=5)
                ))

            fig.update_layout(
                title=f"{title} - 2D Embedding",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                width=800,
                height=600
            )

        elif n_dims == 3:
            # 3D scatter plot
            if labels is not None:
                unique_labels = np.unique(labels)
                fig = go.Figure()
                for label in unique_labels:
                    mask = labels == label
                    fig.add_trace(go.Scatter3d(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        z=embedding[mask, 2],
                        mode='markers',
                        name=str(label),
                        marker=dict(size=3)
                    ))
            else:
                fig = go.Figure(data=go.Scatter3d(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    z=embedding[:, 2],
                    mode='markers',
                    marker=dict(size=3)
                ))

            fig.update_layout(
                title=f"{title} - 3D Embedding",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3"
                ),
                width=900,
                height=700
            )

        fig.show()

    @staticmethod
    def available_sklearn() -> Dict[str, Tuple[str, ...]]:
        loads = tuple(a for a in dir(skds) if a.startswith("load_") and callable(getattr(skds, a)))
        fetches = tuple(a for a in dir(skds) if a.startswith("fetch_") and callable(getattr(skds, a)))
        makes = tuple(a for a in dir(skds) if a.startswith("make_") and callable(getattr(skds, a)))
        return {"load": loads, "fetch": fetches, "make": makes}

    @staticmethod
    def available_cytof() -> Tuple[str, ...]:
        return tuple(_CYTOF_REGISTRY.keys())
