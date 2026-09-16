/**
 * psr.cpp — Implementation of Probabilistic Sharpe Ratio.
 * See psr.h for full documentation.
 */

#include "psr.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace overfitting {

// ── Standard normal helpers ───────────────────────────────────────────────────

double norm_cdf(double x) {
    // Φ(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double norm_pdf(double x) {
    static const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    return inv_sqrt2pi * std::exp(-0.5 * x * x);
}

// ── Return statistics ─────────────────────────────────────────────────────────

double mean(const std::vector<double>& r) {
    if (r.empty()) return 0.0;
    return std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
}

double stddev(const std::vector<double>& r, double mu) {
    int n = static_cast<int>(r.size());
    if (n < 2) return 0.0;
    if (std::isnan(mu)) mu = mean(r);
    double sum_sq = 0.0;
    for (double x : r) {
        double d = x - mu;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / static_cast<double>(n - 1));
}

double skewness(const std::vector<double>& r) {
    int n = static_cast<int>(r.size());
    if (n < 3) return 0.0;
    double mu  = mean(r);
    double sig = stddev(r, mu);
    if (sig == 0.0) return 0.0;
    double sum3 = 0.0;
    for (double x : r) {
        double d = (x - mu) / sig;
        sum3 += d * d * d;
    }
    // Fisher's unbiased: n / ((n-1)*(n-2)) * sum of z^3
    double nd = static_cast<double>(n);
    return (nd / ((nd - 1.0) * (nd - 2.0))) * sum3;
}

double excess_kurtosis(const std::vector<double>& r) {
    int n = static_cast<int>(r.size());
    if (n < 4) return 0.0;
    double mu  = mean(r);
    double sig = stddev(r, mu);
    if (sig == 0.0) return 0.0;
    double sum4 = 0.0;
    for (double x : r) {
        double d = (x - mu) / sig;
        sum4 += d * d * d * d;
    }
    double nd = static_cast<double>(n);
    // Fisher-corrected excess kurtosis
    double kurt_raw = (nd * (nd + 1.0)) / ((nd - 1.0) * (nd - 2.0) * (nd - 3.0)) * sum4;
    double correction = 3.0 * (nd - 1.0) * (nd - 1.0) / ((nd - 2.0) * (nd - 3.0));
    return kurt_raw - correction;
}

// ── Sharpe ratio ──────────────────────────────────────────────────────────────

double sharpe_ratio_period(const std::vector<double>& r) {
    double mu  = mean(r);
    double sig = stddev(r);
    if (sig == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return mu / sig;
}

double sharpe_ratio(const std::vector<double>& r, int freq) {
    return sharpe_ratio_period(r) * std::sqrt(static_cast<double>(freq));
}

// ── PSR ───────────────────────────────────────────────────────────────────────

double psr_from_stats(double sr_hat, int T, double skew, double kurt,
                      double sr_star, int freq) {
    if (T < 2) return std::numeric_limits<double>::quiet_NaN();

    // Convert annualised SR to per-period scale for the variance formula
    double sr_p = sr_hat / std::sqrt(static_cast<double>(freq));

    // Variance of the SR estimator (per-period, then annualise the z-score)
    // Var(SR) ≈ (1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1)
    double var_sr = (1.0 - skew * sr_p + (kurt / 4.0) * sr_p * sr_p)
                    / static_cast<double>(T - 1);

    if (var_sr <= 0.0) {
        // Degenerate: fall back to normal approximation
        var_sr = 1.0 / static_cast<double>(T - 1);
    }

    // SR* must also be in per-period units for the z-score
    double sr_star_p = sr_star / std::sqrt(static_cast<double>(freq));

    double z = (sr_p - sr_star_p) / std::sqrt(var_sr);
    return norm_cdf(z);
}

double psr(const std::vector<double>& r, double sr_star, int freq) {
    int T = static_cast<int>(r.size());
    double sr_hat = sharpe_ratio(r, freq);
    double skew   = skewness(r);
    double kurt   = excess_kurtosis(r);
    return psr_from_stats(sr_hat, T, skew, kurt, sr_star, freq);
}

// ── Full result ───────────────────────────────────────────────────────────────

PSRResult compute_psr(const std::vector<double>& r, double sr_star, int freq) {
    PSRResult res;
    res.T        = static_cast<int>(r.size());
    res.sr_star  = sr_star;
    res.sr_hat   = sharpe_ratio(r, freq);
    res.skew     = skewness(r);
    res.kurt     = excess_kurtosis(r);
    res.psr_value = psr_from_stats(res.sr_hat, res.T, res.skew, res.kurt, sr_star, freq);

    // Variance of SR (per-period)
    double sr_p = res.sr_hat / std::sqrt(static_cast<double>(freq));
    res.variance_sr = (1.0 - res.skew * sr_p + (res.kurt / 4.0) * sr_p * sr_p)
                      / static_cast<double>(std::max(res.T - 1, 1));
    return res;
}

void print_psr_result(const PSRResult& r, const std::string& label) {
    std::cout << std::fixed << std::setprecision(4);
    if (!label.empty()) std::cout << "=== " << label << " ===\n";
    std::cout << "  Observations (T) : " << r.T          << "\n"
              << "  SR_hat (annual)  : " << r.sr_hat     << "\n"
              << "  SR*  (benchmark) : " << r.sr_star    << "\n"
              << "  Skewness         : " << r.skew       << "\n"
              << "  Excess kurtosis  : " << r.kurt       << "\n"
              << "  Var(SR)          : " << r.variance_sr << "\n"
              << "  PSR              : " << r.psr_value  << "\n\n";
}

} // namespace overfitting
