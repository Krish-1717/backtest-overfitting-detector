#pragma once
/**
 * stats.h - Core statistical functions for overfitting detection
 *
 * Implements:
 *   PSR: Probabilistic Sharpe Ratio  [Bailey & Lopez de Prado, 2012]
 *   DSR: Deflated Sharpe Ratio       [Bailey & Lopez de Prado, 2014]
 *
 * No external dependencies - C++17 std library only.
 */

#include <vector>
#include <cmath>
#include <numeric>

namespace detector {

// Basic moments
double mean(const std::vector<double>& r);
double variance(const std::vector<double>& r);    // sample (ddof=1)
double std_dev(const std::vector<double>& r);
double skewness(const std::vector<double>& r);
double excess_kurtosis(const std::vector<double>& r);

/**
 * Annualised Sharpe ratio.
 * @param r          daily return series
 * @param risk_free  daily risk-free rate (default 0)
 * @param ann_factor annualisation factor (default 252)
 */
double sharpe_ratio(const std::vector<double>& r,
                    double risk_free = 0.0, double ann_factor = 252.0);

// Normal CDF / inverse CDF
double normal_cdf(double x);   // Phi(x), max error 7.5e-8
double normal_icdf(double p);  // Phi^{-1}(p)

/**
 * PSR = Phi[(SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt/4+0.25)*SR^2)]
 *
 * Accounts for non-normality (fat tails, skewness) of return distribution.
 * Returns P(true SR > sr_star) in [0, 1].
 */
double psr(const std::vector<double>& r, double sr_star = 0.5,
           double ann_factor = 252.0);

/**
 * DSR adjusts PSR for the expected max SR when testing N strategies.
 *
 * E[max SR] approx (1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e))
 * where gamma = Euler-Mascheroni constant (~0.5772)
 *
 * Returns P(true SR of selected strategy > multiple-testing-adjusted benchmark).
 */
double dsr(const std::vector<double>& r, int n_trials,
           double sr_star_base = 0.5, double ann_factor = 252.0);

double expected_max_sharpe(int n_trials, double sr_base, int n_obs);

} // namespace detector
