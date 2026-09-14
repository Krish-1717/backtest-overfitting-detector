// clustering.h -- Correlation-based clustering for effective-N estimation.
// Day 2: Groups strategies by return correlation, then counts clusters as
// the "effective number" of independent trials for PSR/DSR corrections.

#pragma once
#ifndef BOPT_CLUSTERING_H
#define BOPT_CLUSTERING_H

#include <vector>
#include <string>
#include <cstddef>

namespace bopt {

// ── Correlation matrix ────────────────────────────────────────────────────────

/**
 * Compute the T×T Pearson correlation matrix from a column-major return matrix.
 * @param cols  Column-major return data: cols[strategy][period]
 * @returns     Symmetric n_strategies × n_strategies correlation matrix (row-major)
 */
std::vector<std::vector<double>> pearson_correlation(
    const std::vector<std::vector<double>>& cols
);

// ── Clustering ────────────────────────────────────────────────────────────────

/**
 * Result of a correlation-based clustering pass.
 */
struct ClusterResult {
    std::vector<int> labels;          ///< Cluster label per strategy (0-indexed)
    int              n_clusters;      ///< Number of distinct clusters
    double           effective_n;     ///< Effective independent trials estimate
    double           threshold;       ///< Correlation threshold used
};

/**
 * Greedy single-linkage clustering on a correlation matrix.
 *
 * Two strategies are merged into the same cluster when their pairwise
 * Pearson correlation exceeds @p threshold.  This mirrors the approach in
 * Bailey & López de Prado (2014) for computing Effective N.
 *
 * @param corr       n×n correlation matrix (from pearson_correlation)
 * @param threshold  Merge strategies with |corr| >= threshold (default 0.7)
 * @returns          ClusterResult with labels and effective_n
 */
ClusterResult cluster_strategies(
    const std::vector<std::vector<double>>& corr,
    double threshold = 0.7
);

// ── Effective-N ───────────────────────────────────────────────────────────────

/**
 * Compute effective number of independent trials from cluster labels.
 *
 * Effective N = n_clusters (one independent draw per cluster).
 * This is a conservative lower bound — ignores within-cluster structure.
 *
 * @param labels      Cluster assignment per strategy
 * @param n_clusters  Total distinct cluster count
 * @returns           Effective N as double for downstream PSR calculations
 */
double effective_n(const std::vector<int>& labels, int n_clusters);

// ── Summary ───────────────────────────────────────────────────────────────────

/**
 * Print a human-readable summary of clustering results to stdout.
 * @param result  ClusterResult from cluster_strategies()
 * @param names   Optional strategy names (used in output if provided)
 */
void print_cluster_summary(
    const ClusterResult& result,
    const std::vector<std::string>& names = {}
);

} // namespace bopt

#endif // BOPT_CLUSTERING_H
