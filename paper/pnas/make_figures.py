"""Generate the three main figures for the paper from saved result files.

Outputs:
  figures/fig1_noise_sweep.pdf   — β₁ counts vs noise σ on figure-8
  figures/fig2_pareto_covertype.pdf — kNN vs topo_err Pareto front
  figures/fig3_arxiv_layouts.pdf — DiRe vs cuML UMAP 2D scatters on 723K arXiv papers

All data pulled from files that already exist on disk:
  /tmp/ripser_noise_sweep.json
  /tmp/optuna_topology_covertype.json
  /home/igor/devel/dire-rapids-arxiv/data/{dire,umap}_layout_n16_d2.npy
  /home/igor/devel/dire-rapids-arxiv/data/meta.parquet
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Figure 1: noise sweep on figure-8
# ──────────────────────────────────────────────────────────────────────────
def make_fig1():
    with open("/tmp/ripser_noise_sweep.json") as f:
        data = json.load(f)

    manifold = "figure-8"
    fig, ax = plt.subplots(figsize=(3.3, 2.4))

    # Extract σ levels (sorted) and per-method β₁ means across seeds.
    per_noise = data[manifold]
    sigmas = sorted(float(s) for s in per_noise.keys())

    method_colors = {
        "original": ("tab:gray", "o", "noisy sample"),
        "DiRe-pca": ("tab:green", "s", "DiRe-pca"),
        "DiRe-spec": ("tab:olive", "^", "DiRe-spectral"),
        "cuML-UMAP": ("tab:orange", "D", "cuML UMAP"),
    }

    for method, (color, marker, label) in method_colors.items():
        means, stds = [], []
        for s in sigmas:
            vals = per_noise[str(s)][method]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.errorbar(sigmas, means, yerr=stds, marker=marker, color=color,
                    label=label, capsize=2, linewidth=1.1, markersize=4)

    ax.axhline(2.0, ls="--", color="k", lw=0.7, alpha=0.6,
               label=r"theoretical $\beta_1 = 2$")
    ax.set_xlabel(r"sampling noise $\sigma$")
    ax.set_ylabel(r"$\beta_1$ significant-bar count")
    ax.set_yscale("log")
    ax.set_ylim(1, 60)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(r"figure-8 ($\beta_1 = 2$): method response to noise")

    outpath = os.path.join(FIG_DIR, "fig1_noise_sweep.pdf")
    fig.savefig(outpath)
    print("wrote", outpath, flush=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 2: Pareto front on covertype
# ──────────────────────────────────────────────────────────────────────────
def make_fig2():
    with open("/tmp/optuna_topology_covertype.json") as f:
        data = json.load(f)

    all_trials = data["all"]
    pareto = data["pareto"]
    baselines = data["baselines"]

    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    # All trials: grey cloud
    xs = [t["values"][1] for t in all_trials]  # topo_err
    ys = [t["values"][0] for t in all_trials]  # kNN
    ax.scatter(xs, ys, s=12, c="lightgray", alpha=0.8, label="all trials",
               edgecolors="none")

    # Pareto front
    px = [t["values"][1] for t in pareto]
    py = [t["values"][0] for t in pareto]
    order = np.argsort(px)
    px = np.array(px)[order]
    py = np.array(py)[order]
    ax.plot(px, py, "-", color="tab:red", lw=1.0, alpha=0.6)
    ax.scatter(px, py, s=40, c="tab:red", zorder=3, label="DiRe Pareto",
               edgecolors="black", linewidths=0.4)

    # Baselines
    bl_styles = {
        "DiRe-pca-default":  ("tab:green", "^", "DiRe-pca (default)"),
        "DiRe-spec-default": ("tab:olive", "s", "DiRe-spec (default)"),
        "cuML-UMAP":         ("tab:blue",  "D", "cuML UMAP"),
    }
    for name, (color, marker, label) in bl_styles.items():
        v = baselines.get(name)
        if v is None:
            continue
        ax.scatter(v[1], v[0], s=60, c=color, marker=marker, zorder=4,
                   label=label, edgecolors="black", linewidths=0.4)

    # Shade the dominance region of UMAP
    umap = baselines.get("cuML-UMAP")
    if umap is not None:
        knn_u, topo_u = umap[0], umap[1]
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.axhspan(knn_u, ylim[1], xmin=0, xmax=(topo_u - ax.get_xlim()[0])
                   / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                   color="tab:red", alpha=0.07)
        ax.text(0.2, knn_u + 0.003, "dominates UMAP", fontsize=6.5,
                color="tab:red", alpha=0.8)

    ax.set_xlabel("topology error (lower = better)")
    ax.set_ylabel("2-D $k$NN accuracy (higher = better)")
    ax.set_title("covertype (581K × 54)")
    ax.legend(frameon=False, loc="lower right", ncol=1)

    outpath = os.path.join(FIG_DIR, "fig2_pareto_covertype.pdf")
    fig.savefig(outpath)
    print("wrote", outpath, flush=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 3: arXiv corpus 2-D layouts, DiRe vs UMAP
# ──────────────────────────────────────────────────────────────────────────
def make_fig3():
    base = "/home/igor/devel/dire-rapids-arxiv/data"
    dire_xy = np.load(f"{base}/dire_layout_n16_d2.npy")
    umap_xy = np.load(f"{base}/umap_layout_n16_d2.npy")
    print(f"  dire shape {dire_xy.shape}  umap shape {umap_xy.shape}", flush=True)

    import pandas as pd
    meta = pd.read_parquet(f"{base}/meta.parquet",
                           columns=["primary_category"])
    primary = meta["primary_category"].to_numpy()

    # Collapse to coarse categories for legibility: keep top-N, rest -> "other"
    top_cats, counts = np.unique(primary, return_counts=True)
    order = np.argsort(-counts)
    keep = set(top_cats[order][:15])
    simplified = np.array(
        [c if c in keep else "other" for c in primary], dtype=object)

    # Stratified subsample per category for scatter legibility
    rng = np.random.default_rng(42)
    max_total = 60_000
    cats = sorted(set(simplified))
    per_cat = max_total // len(cats)
    keep_idx = []
    for c in cats:
        w = np.where(simplified == c)[0]
        if len(w) > per_cat:
            w = rng.choice(w, per_cat, replace=False)
        keep_idx.append(w)
    idx = np.concatenate(keep_idx)
    rng.shuffle(idx)

    dire_sub = dire_xy[idx]
    umap_sub = umap_xy[idx]
    lab_sub = simplified[idx]

    cmap = plt.get_cmap("tab20")
    cat_to_color = {c: cmap(i % 20) for i, c in enumerate(cats)}
    colors = np.array([cat_to_color[c] for c in lab_sub])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                    gridspec_kw={"wspace": 0.06})
    for ax, xy, title in [(ax1, dire_sub, "DiRe"),
                           (ax2, umap_sub, "cuML UMAP")]:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.6, alpha=0.5,
                   linewidths=0)
        ax.set_title(f"{title}  (n={len(idx):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "datalim")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # Legend strip under the plots
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cat_to_color[c], markersize=5, label=c)
               for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.03))

    outpath = os.path.join(FIG_DIR, "fig3_arxiv_layouts.pdf")
    fig.savefig(outpath)
    print("wrote", outpath, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    make_fig1()
    make_fig2()
    make_fig3()
    print("done", flush=True)
