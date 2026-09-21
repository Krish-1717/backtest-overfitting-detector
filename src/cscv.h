/**
 * src/cscv.h -- Combinatorially Symmetric Cross-Validation (CSCV)
 *
 * Day 9 Commit 1: Header for CSCV / PBO estimation.
 * Bailey & Lopez de Prado (2012).
 */
#pragma once
#include <vector>
#include <cstddef>

namespace backtest {

struct Matrix {
    std::vector<double> data;
    std::size_t rows{0}, cols{0};
    double  at(std::size_t r, std::size_t c) const { return data[r*cols+c]; }
    double& at(std::size_t r, std::size_t c)       { return data[r*cols+c]; }
};

struct CscvTrial {
    int    is_best;
    double is_sharpe;
    double oos_sharpe;
    double lambda_c;
};

struct CscvResult {
    double pbo;
    double expected_oos_sharpe;
    double rank_deficiency;
    std::size_t n_trials;
    std::vector<CscvTrial> trials;
};

/** Split TÃN returns matrix into S equal time slices. */
std::vector<Matrix> cscv_partition(const Matrix& returns, std::size_t S);

/** Annualised Sharpe (252 days). Returns NaN if std < 1e-12. */
double cscv_sharpe(const double* rets, std::size_t n);

/** Run all C(S, S/2) combinations; returns PBO and related stats. */
CscvResult cscv_pbo(const Matrix& returns, std::size_t S = 16,
                    bool keep_all = false);

/** Estimate PBO from lambda-scores as fraction < 0. */
double pbo_from_logit(const std::vector<double>& lambda_scores);

} // namespace backtest
