// clustering.cpp -- Implementation of correlation clustering for effective-N.
// Day 2: Provides greedy single-linkage clustering over a Pearson correlation
// matrix.  The number of clusters approximates the effective number of
// independent backtest trials (Bailey & Lopez de Prado 2014).

#include "clustering.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace bopt {

// ── Pearson correlation ────────────────────────────────────────────────────────

std::vector<std::vector<double>> pearson_correlation(
    const std::vector<std::vector<double>>& cols
) {
    const std::size_t n = cols.size();
    std::vector<std::vector<double>> corr(n, std::vector<double>(n, 0.0));

    if (n == 0) return corr;

    // Pre-compute means and standard deviations per strategy
    std::vector<double> means(n), stds(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& v = cols[i];
        double sum = 0.0;
        for (double x : v) sum += x;
        means[i] = sum / static_cast<double>(v.size());

        double var = 0.0;
        for (double x : v) {
            double d = x - means[i];
            var += d * d;
        }
        stds[i] = std::sqrt(var / static_cast<double>(v.size()));
    }

    // Fill correlation matrix
    for (std::size_t i = 0; i < n; ++i) {
        corr[i][i] = 1.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            if (stds[i] < 1e-12 || stds[j] < 1e-12) {
                corr[i][j] = corr[j][i] = 0.0;
                continue;
            }
            const auto& vi = cols[i];
            const auto& vj = cols[j];
            std::size_t T = std::min(vi.size(), vj.size());
            double cov = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                cov += (vi[t] - means[i]) * (vj[t] - means[j]);
            }
            cov /= static_cast<double>(T);
            double r = cov / (stds[i] * stds[j]);
            // Clamp to [-1, 1] for numerical safety
            r = std::max(-1.0, std::min(1.0, r));
            corr[i][j] = corr[j][i] = r;
        }
    }
    return corr;
}

// ── Greedy single-linkage clustering ──────────────────────────────────────────

ClusterResult cluster_strategies(
    const std::vector<std::vector<double>>& corr,
    double threshold
) {
    const std::size_t n = corr.size();
    ClusterResult result;
    result.threshold = threshold;
    result.labels.assign(n, -1);
    result.n_clusters = 0;

    for (std::size_t i = 0; i < n; ++i) {
        if (result.labels[i] != -1) continue;  // already assigned
        int cluster_id = result.n_clusters++;
        result.labels[i] = cluster_id;

        // Assign all strategies correlated with i (transitively via single-link)
        for (std::size_t j = i + 1; j < n; ++j) {
            if (result.labels[j] != -1) continue;
            // Check if j is close to ANY member already in this cluster
            bool merge = false;
            for (std::size_t k = 0; k < n && !merge; ++k) {
                if (result.labels[k] == cluster_id) {
                    if (std::abs(corr[k][j]) >= threshold) {
                        merge = true;
                    }
                }
            }
            if (merge) {
                result.labels[j] = cluster_id;
            }
        }
    }

    result.effective_n = effective_n(result.labels, result.n_clusters);
    return result;
}

// ── Effective-N ───────────────────────────────────────────────────────────────

double effective_n(const std::vector<int>& labels, int n_clusters) {
    // Conservative estimate: one independent trial per cluster
    return static_cast<double>(n_clusters);
}

// ── Summary printer ───────────────────────────────────────────────────────────

void print_cluster_summary(
    const ClusterResult& result,
    const std::vector<std::string>& names
) {
    std::cout << "\n=== Cluster Summary ===\n";
    std::cout << "  Threshold   : " << std::fixed << std::setprecision(2)
              << result.threshold << "\n";
    std::cout << "  n_strategies: " << result.labels.size() << "\n";
    std::cout << "  n_clusters  : " << result.n_clusters << "\n";
    std::cout << "  effective_N : " << std::setprecision(1)
              << result.effective_n << "\n\n";

    for (int c = 0; c < result.n_clusters; ++c) {
        std::cout << "  Cluster " << c << ": [";
        bool first = true;
        for (std::size_t i = 0; i < result.labels.size(); ++i) {
            if (result.labels[i] == c) {
                if (!first) std::cout << ", ";
                if (!names.empty() && i < names.size()) {
                    std::cout << names[i];
                } else {
                    std::cout << i;
                }
                first = false;
            }
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

} // namespace bopt
