"""
scripts/full_analysis.py -- End-to-end overfitting analysis pipeline
Day 10 Commit 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pbo import compute_pbo_python
from scripts.compare import run_comparison


def generate_synthetic_returns(T=504, N=20, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, (T, 1))
    noise = rng.normal(0, 0.008, (T, N))
    returns = base + noise
    return returns


def sharpe(rets, ann=252):
    m, s = rets.mean(), rets.std(ddof=1)
    if s < 1e-12:
        return float("nan")
    return float(m / s * np.sqrt(ann))


def run_cscv_python(returns, S=16):
    """Pure-Python CSCV for the analysis script."""
    T, N = returns.shape
    from itertools import combinations as itercombs
    indices = list(range(S))
    half = S // 2
    base = T // S
    extra = T % S
    sizes = [base + (1 if i < extra else 0) for i in range(S)]
    slices = []
    row = 0
    for sz in sizes:
        slices.append(returns[row:row+sz])
        row += sz
    
    lambdas = []
    oos_sharpes = []
    for is_idx in itercombs(indices, half):
        oos_idx = [i for i in indices if i not in is_idx]
        is_mat = np.vstack([slices[i] for i in is_idx])
        oos_mat = np.vstack([slices[i] for i in oos_idx])
        is_sr = np.array([sharpe(is_mat[:, c]) for c in range(N)])
        oos_sr = np.array([sharpe(oos_mat[:, c]) for c in range(N)])
        best = int(np.nanargmax(is_sr))
        oos_best = float(oos_sr[best])
        sorted_oos = np.sort(oos_sr[np.isfinite(oos_sr)])
        rank = int(np.searchsorted(sorted_oos, oos_best, side='left')) + 1
        rel = np.clip(rank / (N + 1), 1e-9, 1 - 1e-9)
        lc = float(np.log(rel / (1 - rel)))
        lambdas.append(lc)
        if np.isfinite(oos_best):
            oos_sharpes.append(oos_best)
    
    pbo = sum(1 for l in lambdas if np.isfinite(l) and l < 0) / len(lambdas)
    exp_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    return {"pbo": pbo, "expected_oos_sharpe": exp_oos, "n_trials": len(lambdas)}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Full backtest overfitting analysis")
    parser.add_argument("--T", type=int, default=504, help="Number of time steps")
    parser.add_argument("--N", type=int, default=20, help="Number of strategies")
    parser.add_argument("--S", type=int, default=16, help="CSCV partitions (even)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--n-clusters", type=int, default=4, help="Clusters for comparison")
    args = parser.parse_args()

    returns = generate_synthetic_returns(T=args.T, N=args.N, seed=args.seed)

    print_section("1. DATA SUMMARY")
    print(f"  Shape: {returns.shape}  (T={args.T} days, N={args.N} strategies)")
    sr_all = [sharpe(returns[:, c]) for c in range(args.N)]
    print(f"  Sharpe (mean Â± std): {np.nanmean(sr_all):.3f} Â± {np.nanstd(sr_all):.3f}")
    print(f"  Best IS Sharpe: {np.nanmax(sr_all):.3f}")

    print_section("2. CSCV / PBO ANALYSIS")
    res = run_cscv_python(returns, S=args.S)
    print(f"  PBO:               {res['pbo']:.4f}")
    print(f"  E[OOS Sharpe]:     {res['expected_oos_sharpe']:.4f}")
    print(f"  Num trials:        {res['n_trials']}")
    overfitting_risk = "HIGH" if res["pbo"] > 0.5 else "MODERATE" if res["pbo"] > 0.25 else "LOW"
    print(f"  Overfitting risk:  {overfitting_risk}")

    print_section("3. STRATEGY CLUSTERING")
    try:
        comparison = run_comparison(returns, n_clusters=args.n_clusters)
        print(f"  K-means silhouette:      {comparison['kmeans']['silhouette']:.4f}")
        print(f"  Hierarchical silhouette: {comparison['hierarchical']['silhouette']:.4f}")
        winner = "K-means" if comparison["kmeans"]["silhouette"] > comparison["hierarchical"]["silhouette"] else "Hierarchical"
        print(f"  Better clustering:       {winner}")
    except Exception as e:
        print(f"  Clustering unavailable: {e}")
        comparison = {}

    print_section("4. SUMMARY")
    best_strat = int(np.nanargmax(sr_all))
    print(f"  Best strategy (IS):       #{best_strat}  SR={sr_all[best_strat]:.3f}")
    print(f"  PBO (prob of overfitting): {res['pbo']:.2%}")
    print(f"  Recommendation: {'Select strategy cautiously â high overfitting risk' if res['pbo'] > 0.5 else 'Strategy selection appears reasonably robust'}")

    if args.json:
        out = {
            "data": {"T": args.T, "N": args.N},
            "cscv": res,
            "best_strategy": {"index": best_strat, "is_sharpe": float(sr_all[best_strat])},
        }
        print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
