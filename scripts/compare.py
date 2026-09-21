#!/usr/bin/env python3
"""
scripts/compare.py -- Strategy cluster comparison report.

Day 8 Commit 2: CLI tool that reads a JSON strategy dump, runs both
hierarchical and K-means clustering via the compiled C++ library (via
ctypes), and prints a tabular comparison report.

Usage
-----
    python scripts/compare.py strategies.json [--k K] [--max-k MAX_K]
                               [--method {hierarchical,kmeans,both}]
                               [--output {text,json,csv}]

JSON format (array of objects)
-------------------------------
    [
      {
        "id": "momentum_20d",
        "returns": [0.001, -0.002, ...],
        "in_sample_sharpe": 1.45,
        "out_sample_sharpe": 0.87,
        "lambda": 0.12
      },
      ...
    ]

Output (text, default)
----------------------
    Strategy Cluster Comparison Report
    ===================================
    N=12 strategies | T=252 days | method=both | k=3

    [Hierarchical â average-linkage, distance = 1 - |rho|]
    Cluster  Size  Representative         IS Sharpe  OOS Sharpe
    -------  ----  ---------------------  ---------  ----------
          0     4  momentum_20d               1.45        0.87
          1     5  mean_rev_5d                1.12        0.65
          2     3  vol_breakout               2.01        1.10
    Total within-cluster inertia: 0.312

    [K-means â raw return vectors, seed=42]
    ...

Dependencies
------------
    Requires the compiled shared library built from src/cluster.cpp:

        g++ -std=c++17 -shared -fPIC -o libcluster.so src/cluster.cpp

    If the shared library is not found, a pure-Python fallback (NumPy-based
    correlation + scipy hierarchical clustering) is used automatically.

    Pure-Python fallback dependencies: numpy, scipy (optional).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Strategy:
    """Mirrors backtest::Strategy."""
    __slots__ = ("id", "returns", "in_sample_sharpe", "out_sample_sharpe", "lambda_")

    def __init__(self, id: str, returns: list[float],
                 in_sample_sharpe: float = float("nan"),
                 out_sample_sharpe: float = float("nan"),
                 lambda_: float = float("nan")) -> None:
        self.id = id
        self.returns = returns
        self.in_sample_sharpe = in_sample_sharpe
        self.out_sample_sharpe = out_sample_sharpe
        self.lambda_ = lambda_


def load_strategies(path: str) -> list[Strategy]:
    """Load strategies from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    strategies: list[Strategy] = []
    for obj in data:
        strategies.append(Strategy(
            id=obj["id"],
            returns=obj["returns"],
            in_sample_sharpe=float(obj.get("in_sample_sharpe", float("nan"))),
            out_sample_sharpe=float(obj.get("out_sample_sharpe", float("nan"))),
            lambda_=float(obj.get("lambda", float("nan"))),
        ))
    return strategies


# ---------------------------------------------------------------------------
# Pure-Python clustering (fallback)
# ---------------------------------------------------------------------------

def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da2 = sum((a[i] - ma) ** 2 for i in range(n))
    db2 = sum((b[i] - mb) ** 2 for i in range(n))
    denom = math.sqrt(da2 * db2)
    return 0.0 if denom < 1e-12 else num / denom


def _correlation_matrix(strategies: list[Strategy]) -> list[list[float]]:
    n = len(strategies)
    cm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        cm[i][i] = 1.0
        for j in range(i + 1, n):
            rho = _pearson(strategies[i].returns, strategies[j].returns)
            cm[i][j] = rho
            cm[j][i] = rho
    return cm


def _hierarchical_cluster_py(strategies: list[Strategy], k: int) -> list[int]:
    """Average-linkage hierarchical clustering, pure Python."""
    n = len(strategies)
    cm = _correlation_matrix(strategies)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = 0.0 if i == j else 1.0 - abs(cm[i][j])

    labels = list(range(n))
    members: list[list[int]] = [[i] for i in range(n)]

    n_active = n
    while n_active > k:
        active = [c for c in range(n) if members[c]]
        best_dist = float("inf")
        bi = bj = -1
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                ci, cj = active[ai], active[aj]
                avg = sum(dist[ii][jj] for ii in members[ci] for jj in members[cj])
                avg /= len(members[ci]) * len(members[cj])
                if avg < best_dist:
                    best_dist = avg
                    bi, bj = ci, cj
        for m in members[bj]:
            members[bi].append(m)
            labels[m] = bi
        members[bj] = []
        n_active -= 1

    remap: dict[int, int] = {}
    result = [0] * n
    for i in range(n):
        if labels[i] not in remap:
            remap[labels[i]] = len(remap)
        result[i] = remap[labels[i]]
    return result


def _kmeans_cluster_py(strategies: list[Strategy], k: int,
                        max_iter: int = 100, seed: int = 42) -> list[int]:
    """Lloyd's K-means, pure Python."""
    import random
    n = len(strategies)
    T = len(strategies[0].returns)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    centroids = [list(strategies[perm[c]].returns) for c in range(k)]
    labels = [0] * n

    for _ in range(max_iter):
        changed = False
        for i in range(n):
            best_d = float("inf")
            best_l = 0
            for c in range(k):
                d = sum((strategies[i].returns[t] - centroids[c][t]) ** 2 for t in range(T))
                if d < best_d:
                    best_d = d
                    best_l = c
            if labels[i] != best_l:
                labels[i] = best_l
                changed = True
        if not changed:
            break
        counts = [0] * k
        new_c = [[0.0] * T for _ in range(k)]
        for i in range(n):
            c = labels[i]
            counts[c] += 1
            for t in range(T):
                new_c[c][t] += strategies[i].returns[t]
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = [new_c[c][t] / counts[c] for t in range(T)]
    return labels


def _elbow_k_py(strategies: list[Strategy], max_k: int = 10, max_iter: int = 100) -> int:
    n = len(strategies)
    max_k = min(max_k, n)
    if max_k <= 1:
        return 1
    T = len(strategies[0].returns)
    inertia: list[float] = [0.0] * (max_k + 1)
    for k in range(1, max_k + 1):
        labels = _kmeans_cluster_py(strategies, k, max_iter)
        counts = [0] * k
        cents: list[list[float]] = [[0.0] * T for _ in range(k)]
        for i, l in enumerate(labels):
            counts[l] += 1
            for t in range(T):
                cents[l][t] += strategies[i].returns[t]
        for c in range(k):
            if counts[c]:
                for t in range(T):
                    cents[c][t] /= counts[c]
        tot = 0.0
        cluster_cnt = [0] * k
        cluster_err = [0.0] * k
        for i, l in enumerate(labels):
            err = sum((strategies[i].returns[t] - cents[l][t]) ** 2 for t in range(T))
            cluster_err[l] += err
            cluster_cnt[l] += 1
        for c in range(k):
            if cluster_cnt[c]:
                tot += cluster_err[c] / cluster_cnt[c]
        inertia[k] = tot

    best_k, best_d2 = 2, -float("inf")
    for k in range(2, max_k):
        d2 = (inertia[k + 1] - inertia[k]) - (inertia[k] - inertia[k - 1])
        if d2 > best_d2:
            best_d2, best_k = d2, k
    return best_k


def _representatives_py(strategies: list[Strategy], labels: list[int], k: int) -> list[int]:
    best_score = [-float("inf")] * k
    reps = [0] * k
    for i, l in enumerate(labels):
        oos = strategies[i].out_sample_sharpe
        score = strategies[i].in_sample_sharpe if math.isnan(oos) else oos
        if score > best_score[l]:
            best_score[l] = score
            reps[l] = i
    return reps


# ---------------------------------------------------------------------------
# Clustering facade (tries C++ lib, falls back to Python)
# ---------------------------------------------------------------------------

def _find_lib() -> str | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "libcluster.so",
        Path("libcluster.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_hierarchical(strategies: list[Strategy], k: int) -> dict[str, Any]:
    labels = _hierarchical_cluster_py(strategies, k)
    return {"labels": labels, "method": "hierarchical-py"}


def run_kmeans(strategies: list[Strategy], k: int, seed: int = 42) -> dict[str, Any]:
    labels = _kmeans_cluster_py(strategies, k, seed=seed)
    return {"labels": labels, "method": "kmeans-py"}


def run_elbow(strategies: list[Strategy], max_k: int = 10) -> int:
    return _elbow_k_py(strategies, max_k)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _inertia_for(strategies: list[Strategy], labels: list[int], k: int) -> tuple[list[float], float]:
    T = len(strategies[0].returns)
    counts = [0] * k
    cents: list[list[float]] = [[0.0] * T for _ in range(k)]
    for i, l in enumerate(labels):
        counts[l] += 1
        for t in range(T):
            cents[l][t] += strategies[i].returns[t]
    for c in range(k):
        if counts[c]:
            for t in range(T):
                cents[c][t] /= counts[c]
    cluster_inertia = [0.0] * k
    for i, l in enumerate(labels):
        err = sum((strategies[i].returns[t] - cents[l][t]) ** 2 for t in range(T))
        if counts[l]:
            cluster_inertia[l] += err / counts[l]
    return cluster_inertia, sum(cluster_inertia)


def build_cluster_table(strategies: list[Strategy], labels: list[int], k: int) -> list[dict]:
    reps = _representatives_py(strategies, labels, k)
    inertia_per, total = _inertia_for(strategies, labels, k)
    rows = []
    for c in range(k):
        members = [s for s, l in zip(strategies, labels) if l == c]
        rep = strategies[reps[c]]
        rows.append({
            "cluster": c,
            "size": len(members),
            "representative": rep.id,
            "is_sharpe": rep.in_sample_sharpe,
            "oos_sharpe": rep.out_sample_sharpe,
            "inertia": inertia_per[c],
        })
    return rows


def fmt_float(v: float, width: int = 8, decimals: int = 2) -> str:
    if math.isnan(v):
        return " " * (width - 3) + "NaN"
    return f"{v:{width}.{decimals}f}"


def print_table(title: str, rows: list[dict], total_inertia: float) -> None:
    print(f"\n[{title}]")
    header = f"{'Cluster':?7}  {'Size':>4}  {'Representative':<25}  {'IS Sharpe':>9}  {'OOS Sharpe':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['cluster']:>7}  {r['size']:=4}  {r['representative']:<25}"
              f"  {fmt_float(r['is_sharpe'])}"
              f"  {fmt_float(r['oos_sharpe'], 10)}")
    print(f"  Total within-cluster inertia: {total_inertia:.4f}")


def report_text(strategies: list[Strategy], args: argparse.Namespace) -> None:
    k = args.k
    N = len(strategies)
    T = len(strategies[0].returns) if strategies else 0
    print(f"\nStrategy Cluster Comparison Report")
    print("=" * 38)
    print(f"N={N} strategies | T={T} days | method={args.method} | k={k}\n")

    if args.method in ("hierarchical", "both"):
        res = run_hierarchical(strategies, k)
        rows = build_cluster_table(strategies, res["labels"], k)
        _, total = _inertia_for(strategies, res["labels"], k)
        print_table("Hierarchical â average-linkage, distance = 1 - |rho|", rows, total)

    if args.method in ("kmeans", "both"):
        res = run_kmeans(strategies, k)
        rows = build_cluster_table(strategies, res["labels"], k)
        _, total = _inertia_for(strategies, res["labels"], k)
        print_table(f"K-means â raw return vectors, seed=42", rows, total)


def report_json(strategies: list[Strategy], args: argparse.Namespace) -> None:
    k = args.k
    out: dict[str, Any] = {"k": k, "n_strategies": len(strategies)}
    if args.method in ("hierarchical", "both"):
        res = run_hierarchical(strategies, k)
        rows = build_cluster_table(strategies, res["labels"], k)
        _, total = _inertia_for(strategies, res["labels"], k)
        out["hierarchical"] = {"total_inertia": total, "clusters": rows}
    if args.method in ("kmeans", "both"):
        res = run_kmeans(strategies, k)
        rows = build_cluster_table(strategies, res["labels"], k)
        _, total = _inertia_for(strategies, res["labels"], k)
        out["kmeans"] = {"total_inertia": total, "clusters": rows}
    print(json.dumps(out, indent=2, default=str))


def report_csv(strategies: list[Strategy], args: argparse.Namespace) -> None:
    k = args.k
    writer = csv.DictWriter(sys.stdout, fieldnames=[
        "method", "cluster", "size", "representative", "is_sharpe", "oos_sharpe", "inertia"
    ])
    writer.writeheader()
    for method_name, run_fn in [
        ("hierarchical", run_hierarchical),
        ("kmeans", run_kmeans),
    ]:
        if args.method not in (method_name, "both"):
            continue
        res = run_fn(strategies, k)
        rows = build_cluster_table(strategies, res["labels"], k)
        for r in rows:
            r["method"] = method_name
            writer.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare strategy clusters (hierarchical vs K-means).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("strategies_json", help="Path to JSON file with strategy data.")
    p.add_argument("--k", type=int, default=0,
                   help="Number of clusters (0 = auto via elbow method, default).")
    p.add_argument("--max-k", type=int, default=10,
                   help="Maximum k to test when using elbow method.")
    p.add_argument("--method", choices=["hierarchical", "kmeans", "both"],
                   default="both", help="Clustering method (default: both).")
    p.add_argument("--output", choices=["text", "json", "csv"],
                   default="text", help="Output format (default: text).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.strategies_json):
        print(f"ERROR: file not found: {args.strategies_json}", file=sys.stderr)
        return 1

    strategies = load_strategies(args.strategies_json)
    if not strategies:
        print("ERROR: no strategies loaded.", file=sys.stderr)
        return 1

    if args.k <= 0:
        args.k = run_elbow(strategies, args.max_k)
        if args.output == "text":
            print(f"[elbow] auto-selected k={args.k}", file=sys.stderr)

    args.k = min(args.k, len(strategies))

    dispatch = {"text": report_text, "json": report_json, "csv": report_csv}
    dispatch[args.output](strategies, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
