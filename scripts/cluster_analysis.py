"""
cluster_analysis.py -- Python visualization of correlation clustering results.

Day 2: Reads a CSV of strategy returns, computes the Pearson correlation
matrix, runs greedy single-linkage clustering, and outputs:
  - A colour-coded correlation heatmap (sorted by cluster)
  - A bar chart of cluster sizes
  - A summary table of effective-N vs n_strategies

Usage:
    python scripts/cluster_analysis.py --file data/returns.csv --threshold 0.7
    python scripts/cluster_analysis.py --demo          # generates synthetic data
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Correlation helpers ────────────────────────────────────────────────────────

def pearson_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute Pearson correlation matrix from a returns DataFrame."""
    return df.corr(method="pearson").values


# ── Greedy single-linkage clustering ──────────────────────────────────────────

def cluster_strategies(corr: np.ndarray, threshold: float = 0.7):
    """
    Assign each strategy to a cluster; strategies with |corr| >= threshold
    to any existing cluster member get merged in (greedy single-linkage).

    Returns
    -------
    labels : np.ndarray of int, shape (n,)
    n_clusters : int
    """
    n = corr.shape[0]
    labels = np.full(n, -1, dtype=int)
    n_clusters = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        cid = n_clusters
        n_clusters += 1
        labels[i] = cid
        for j in range(i + 1, n):
            if labels[j] != -1:
                continue
            if any(abs(corr[k, j]) >= threshold for k in range(n) if labels[k] == cid):
                labels[j] = cid

    return labels, n_clusters


def effective_n(n_clusters: int) -> float:
    """Conservative effective-N: one independent trial per cluster."""
    return float(n_clusters)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr: np.ndarray,
    labels: np.ndarray,
    names: list,
    threshold: float,
    out_path: Path = None,
):
    """Plot a cluster-sorted correlation heatmap."""
    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not installed — skipping heatmap")
        return

    # Sort strategies by cluster
    order = np.argsort(labels)
    corr_sorted = corr[np.ix_(order, order)]
    names_sorted = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), max(6, len(names) * 0.4)))
    im = ax.imshow(corr_sorted, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(names_sorted)))
    ax.set_yticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=90, fontsize=7)
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.set_title(
        f"Strategy Correlation Matrix (cluster-sorted, threshold={threshold:.2f})",
        fontsize=10,
    )

    # Draw cluster boundaries
    labels_sorted = labels[order]
    boundaries = np.where(np.diff(labels_sorted))[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=1.5)
        ax.axvline(b - 0.5, color="white", linewidth=1.5)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved heatmap to {out_path}")
    else:
        plt.show()
    plt.close()


def plot_cluster_sizes(labels: np.ndarray, n_clusters: int, out_path: Path = None):
    """Bar chart of strategies per cluster."""
    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not installed — skipping cluster-size chart")
        return

    sizes = [int((labels == c).sum()) for c in range(n_clusters)]
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 0.5), 4))
    ax.bar(range(n_clusters), sizes, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("# Strategies")
    ax.set_title("Strategies per Cluster")
    ax.axhline(1, color="red", linestyle="--", linewidth=0.8, label="size=1")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved cluster-size chart to {out_path}")
    else:
        plt.show()
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(n_strats: int, n_clusters: int, threshold: float):
    eff_n = effective_n(n_clusters)
    reduction = 1.0 - eff_n / n_strats if n_strats > 0 else 0.0

    print("\n" + "=" * 50)
    print("  Backtest Overfitting Detector — Cluster Analysis")
    print("=" * 50)
    print(f"  n_strategies  : {n_strats}")
    print(f"  threshold     : {threshold:.2f}")
    print(f"  n_clusters    : {n_clusters}")
    print(f"  effective_N   : {eff_n:.1f}")
    print(f"  reduction     : {reduction*100:.1f}%  (vs naively assuming independence)")
    print("=" * 50 + "\n")


# ── Demo data ─────────────────────────────────────────────────────────────────

def make_demo_returns(n_strategies: int = 20, n_periods: int = 252, seed: int = 42):
    """Generate synthetic strategy returns with embedded correlation clusters."""
    rng = np.random.default_rng(seed)
    # 4 latent factors
    n_factors = 4
    factors = rng.normal(0, 0.01, (n_periods, n_factors))
    loadings = rng.uniform(0.3, 0.9, (n_strategies, n_factors))
    # Assign each strategy a dominant factor
    dominant = rng.integers(0, n_factors, n_strategies)
    loadings *= 0.1
    for i, d in enumerate(dominant):
        loadings[i, d] = rng.uniform(0.6, 0.9)

    returns = factors @ loadings.T + rng.normal(0, 0.005, (n_periods, n_strategies))
    cols = [f"S{i:02d}" for i in range(n_strategies)]
    return pd.DataFrame(returns, columns=cols)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Correlation clustering for backtest overfitting detection")
    p.add_argument("--file",      default=None,  help="Path to CSV of strategy returns (rows=periods, cols=strategies)")
    p.add_argument("--threshold", type=float, default=0.7, help="Correlation threshold for merging (default 0.7)")
    p.add_argument("--demo",      action="store_true", help="Use synthetic demo data")
    p.add_argument("--out_dir",   default=None, help="Directory to save plots (default: display interactively)")
    p.add_argument("--n_strats",  type=int, default=20, help="Number of strategies for --demo")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo or args.file is None:
        print(f"Generating demo data ({args.n_strats} strategies, 252 periods) ...")
        df = make_demo_returns(n_strategies=args.n_strats)
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(file_path, index_col=0)
        print(f"Loaded {df.shape[1]} strategies x {df.shape[0]} periods from {file_path}")

    names = list(df.columns)
    corr = pearson_matrix(df)
    labels, n_clusters = cluster_strategies(corr, threshold=args.threshold)

    print_summary(len(names), n_clusters, args.threshold)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_correlation_heatmap(
        corr, labels, names, args.threshold,
        out_path=(out_dir / "correlation_heatmap.png") if out_dir else None,
    )
    plot_cluster_sizes(
        labels, n_clusters,
        out_path=(out_dir / "cluster_sizes.png") if out_dir else None,
    )


if __name__ == "__main__":
    main()
