#pragma once
/**
 * psr.h — Probabilistic Sharpe Ratio (PSR) and related helpers.
 *
 * Day 3: The PSR answers:
 *   "What is the probability that the TRUE Sharpe ratio of this strategy
 *    exceeds a benchmark SR*, given only the observed sample?"
 *
 * Formula (Bailey & López de Prado, 2012):
 *
 *   PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(T-1)
 *               / sqrt(1 - γ₃·SR_hat + (γ₄-1)/4·SR_hat²) )
 *
 * where:
 *   SR_hat = annualised sample Sharpe ratio (per-period, then * sqrt(freq))
 *   T      = number of return observations
 *   γ₃     = skewness of returns
 *   γ₄     = excess kurtosis of returns
 *   Φ      = standard normal CDF
 *   freq   = trading periods per year (252 for daily, 12 for monthly)
 *
 * References:
 *   Bailey, D. H., & López de Prado, M. (2012).
 *   "The Sharpe Ratio Efficient Frontier."
 *   Journal of Risk, 15(2), 3–44.
 */

#ifndef PSR_H
#define PSR_H

#include <vector>
#include <string>

namespace overfitting {

// ── Return statistics ─────────────────────────────────────────────────────────

/**
 * Compute sample mean of a return series.
 */
double mean(const std::vector<double>& returns);

/**
 * Compute sample standard deviation (ddof=1).
 */
double stddev(const std::vector<double>& returns, double mu = std::numeric_limits<double>::quiet_NaN());

/**
 * Compute sample skewness (Fisher's definition, unbiased correction).
 */
double skewness(const std::vector<double>& returns);

/**
 * Compute sample excess kurtosis (Fisher's definition, unbiased correction).
 */
double excess_kurtosis(const std::vector<double>& returns);

// ── Sharpe ratio ──────────────────────────────────────────────────────────────

/**
 * Per-period (non-annualised) sample Sharpe ratio.
 *   SR = mean(returns) / stddev(returns)
 * Returns NaN if stddev == 0.
 */
double sharpe_ratio_period(const std::vector<double>& returns);

/**
 * Annualised sample Sharpe ratio.
 *   SR_annual = SR_period * sqrt(freq)
 * @param freq  Trading periods per year (252=daily, 52=weekly, 12=monthly).
 */
double sharpe_ratio(const std::vector<double>& returns, int freq = 252);

// ── PSR ───────────────────────────────────────────────────────────────────────

/**
 * Probabilistic Sharpe Ratio.
 *
 * @param returns   Vector of per-period returns (e.g. daily P&L / NAV).
 * @param sr_star   Benchmark Sharpe ratio (annualised, default 0).
 * @param freq      Trading periods per year.
 * @returns         Probability in [0,1] that true SR > sr_star.
 */
double psr(const std::vector<double>& returns,
           double sr_star = 0.0,
           int    freq    = 252);

/**
 * PSR given pre-computed statistics (avoids re-computing moments).
 *
 * @param sr_hat    Observed annualised Sharpe ratio.
 * @param T         Number of return observations.
 * @param skew      Sample skewness.
 * @param kurt      Sample excess kurtosis.
 * @param sr_star   Benchmark SR (annualised).
 * @param freq      Periods per year (used to convert sr_hat to per-period).
 */
double psr_from_stats(double sr_hat,
                      int    T,
                      double skew,
                      double kurt,
                      double sr_star = 0.0,
                      int    freq    = 252);

// ── Standard normal helpers ───────────────────────────────────────────────────

/**
 * Standard normal CDF Φ(x) using std::erfc.
 */
double norm_cdf(double x);

/**
 * Standard normal PDF φ(x).
 */
double norm_pdf(double x);

// ── Result struct ─────────────────────────────────────────────────────────────

struct PSRResult {
    double sr_hat;          // observed annualised Sharpe ratio
    double sr_star;         // benchmark SR
    double psr_value;       // PSR = P(true SR > sr_star)
    int    T;               // number of observations
    double skew;
    double kurt;
    double variance_sr;     // variance of SR estimator (for reference)
};

/**
 * Compute full PSR result struct.
 */
PSRResult compute_psr(const std::vector<double>& returns,
                      double sr_star = 0.0,
                      int    freq    = 252);

/**
 * Pretty-print a PSR result.
 */
void print_psr_result(const PSRResult& r, const std::string& label = "");

} // namespace overfitting

#endif // PSR_H
