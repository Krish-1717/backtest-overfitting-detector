"""
scripts/benchmark.py -- Benchmark multiple strategies with PBO and Sharpe comparison
backtest-overfitting-detector Day 11
"""
from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def _sharpe(returns: List[float], ann: int = 252) -> float:
    n = len(returns)
    if n < 2:
        return float("nan")
    mu = sum(returns) / n
    var = sum((r - mu)**2 for r in returns) / (n - 1)
    sigma = math.sqrt(var)
    if sigma < 1e-12:
        return float("nan")
    return mu / sigma * math.sqrt(ann)


def _max_drawdown(returns: List[float]) -> float:
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for r in returns:
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _calmar(returns: List[float], ann: int = 252) -> float:
    mdd = _max_drawdown(returns)
    if mdd < 1e-12:
        return float("nan")
    ann_ret = sum(returns) / len(returns) * ann
    return ann_ret / mdd


def generate_strategy(n: int = 500, sharpe_target: float = 0.5,
                      seed: int = 0) -> List[float]:
    """Generate synthetic daily returns for a strategy with given target Sharpe."""
    rng = random.Random(seed)
    daily_mu = sharpe_target / math.sqrt(252) * 0.01
    daily_sigma = 0.01
    return [rng.gauss(daily_mu, daily_sigma) for _ in range(n)]


def _pbo_python(returns_matrix: List[List[float]], n_partitions: int = 8) -> float:
    """Pure-Python CSCV PBO estimate."""
    from itertools import combinations
    T = len(returns_matrix[0])
    S = n_partitions
    if S % 2 != 0:
        S -= 1
    part_size = T // S
    parts = [list(range(i * part_size, (i + 1) * part_size)) for i in range(S)]
    n_strats = len(returns_matrix)

    def sharpe_on_indices(strat_idx: int, indices: List[int]) -> float:
        rets = [returns_matrix[strat_idx][i] for i in indices]
        return _sharpe(rets)

    lambda_c_list = []
    combo_count = 0
    max_combos = 200  # limit for speed
    for is_parts in combinations(range(S), S // 2):
        if combo_count >= max_combos:
            break
        combo_count += 1
        oos_parts = [p for p in range(S) if p not in is_parts]
        is_idx = [i for p in is_parts for i in parts[p]]
        oos_idx = [i for p in oos_parts for i in parts[p]]
        is_sharpes = [sharpe_on_indices(s, is_idx) for s in range(n_strats)]
        oos_sharpes = [sharpe_on_indices(s, oos_idx) for s in range(n_strats)]
        best_is = max(range(n_strats), key=lambda s: (is_sharpes[s] if not math.isnan(is_sharpes[s]) else -1e9))
        oos_best = oos_sharpes[best_is]
        valid = [s for s in oos_sharpes if not math.isnan(s)]
        if not valid:
            continue
        rank = sum(1 for s in valid if s < oos_best)
        rel_rank = rank / len(valid)
        if rel_rank in (0.0, 1.0):
            rel_rank = max(0.001, min(rel_rank, 0.999))
        lambda_c = math.log(rel_rank / (1 - rel_rank))
        lambda_c_list.append(lambda_c)

    if not lambda_c_list:
        return float("nan")
    return sum(1 for lc in lambda_c_list if lc < 0) / len(lambda_c_list)


def run_benchmark(args):
    rng = random.Random(args.seed)
    strategies = {
        f"strat_{i:02d}": generate_strategy(
            n=args.n, sharpe_target=rng.uniform(-0.5, 1.5), seed=i
        )
        for i in range(args.n_strategies)
    }
    metrics = {}
    for name, rets in strategies.items():
        metrics[name] = {
            "sharpe": round(_sharpe(rets), 4),
            "calmar": round(_calmar(rets), 4) if not math.isnan(_calmar(rets)) else None,
            "max_drawdown": round(_max_drawdown(rets), 6),
            "ann_return": round(sum(rets) / len(rets) * 252, 6),
        }

    returns_matrix = list(strategies.values())
    pbo = _pbo_python(returns_matrix, n_partitions=args.partitions)

    result = {
        "n_strategies": args.n_strategies,
        "n_days": args.n,
        "pbo": round(pbo, 4) if not math.isnan(pbo) else None,
        "best_is_sharpe": max(m["sharpe"] for m in metrics.values()),
        "median_oos_sharpe": sorted(m["sharpe"] for m in metrics.values())[len(metrics)//2],
        "strategy_metrics": metrics,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 65)
    print("  BACKTEST BENCHMARK â Strategy Comparison & PBO")
    print("=" * 65)
    print(f"\n  Strategies: {args.n_strategies}  |  Days: {args.n}  |  Partitions: {args.partitions}")
    print(f"\n  PBO (Probability of Backtest Overfitting): {pbo:.4f}")
    overfitting_risk = "HIGH" if pbo > 0.5 else "MODERATE" if pbo > 0.3 else "LOW"
    print(f"  Overfitting risk: {overfitting_risk}")
    print(f"\n  {'Strategy':<12}  {'Sharpe':>8}  {'Calmar':>8}  {'MaxDD':>8}  {'AnnRet':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, m in sorted(metrics.items(), key=lambda x: -x[1]["sharpe"])[:10]:
        calmar_str = f"{m['calmar']:>8.3f}" if m["calmar"] is not None else f"{'N/A':>8}"
        print(f"  {name:<12}  {m['sharpe']:>8.3f}  {calmar_str}  {m['max_drawdown']:>8.5f}  {m['ann_return']:>8.5f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-strategies", type=int, default=20)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--partitions", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    run_benchmark(p.parse_args())


if __name__ == "__main__":
    main()
