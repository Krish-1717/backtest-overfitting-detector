/**
 * tests/test_cscv.cpp -- Unit tests for CSCV / PBO
 * Day 10 Commit 1
 */
#include "../src/cscv.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace backtest;

static Matrix make_returns(std::size_t T, std::size_t N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(0.0, 0.01);
    Matrix m; m.rows = T; m.cols = N; m.data.resize(T * N);
    for (auto& v : m.data) v = d(rng);
    return m;
}

static void test_sharpe_nan_on_empty() {
    double v = cscv_sharpe(nullptr, 0);
    assert(std::isnan(v));
    std::cout << "PASS: sharpe nan on empty\n";
}

static void test_sharpe_nan_on_zero_std() {
    double r[5] = {0.01, 0.01, 0.01, 0.01, 0.01};
    double v = cscv_sharpe(r, 5);
    // constant series â std=0 â NaN
    assert(std::isnan(v));
    std::cout << "PASS: sharpe nan on zero std\n";
}

static void test_sharpe_positive_for_positive_mean() {
    double r[5] = {0.02, 0.03, 0.01, 0.02, 0.025};
    double v = cscv_sharpe(r, 5);
    assert(std::isfinite(v) && v > 0.0);
    std::cout << "PASS: sharpe positive for positive mean\n";
}

static void test_partition_sizes() {
    Matrix m = make_returns(100, 3);
    auto parts = cscv_partition(m, 4);
    assert(parts.size() == 4);
    std::size_t total = 0;
    for (auto& p : parts) { assert(p.cols == 3); total += p.rows; }
    assert(total == 100);
    std::cout << "PASS: partition sizes sum correctly\n";
}

static void test_partition_rejects_odd_S() {
    Matrix m = make_returns(100, 3);
    bool caught = false;
    try { cscv_partition(m, 3); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
    std::cout << "PASS: partition rejects odd S\n";
}

static void test_pbo_range() {
    Matrix m = make_returns(252, 10);
    auto res = cscv_pbo(m, 4);
    assert(res.pbo >= 0.0 && res.pbo <= 1.0);
    assert(res.n_trials > 0);
    std::cout << "PASS: PBO in [0,1]\n";
}

static void test_pbo_from_logit_all_positive() {
    std::vector<double> lam = {1.0, 2.0, 0.5};
    double pbo = pbo_from_logit(lam);
    assert(std::abs(pbo) < 1e-12);
    std::cout << "PASS: pbo=0 when all lambda>0\n";
}

static void test_pbo_from_logit_all_negative() {
    std::vector<double> lam = {-1.0, -2.0, -0.5};
    double pbo = pbo_from_logit(lam);
    assert(std::abs(pbo - 1.0) < 1e-12);
    std::cout << "PASS: pbo=1 when all lambda<0\n";
}

static void test_keep_all_trials() {
    Matrix m = make_returns(120, 5);
    auto res = cscv_pbo(m, 4, /*keep_all=*/true);
    assert(res.trials.size() == res.n_trials);
    std::cout << "PASS: keep_all stores all trials\n";
}

int main() {
    test_sharpe_nan_on_empty();
    test_sharpe_nan_on_zero_std();
    test_sharpe_positive_for_positive_mean();
    test_partition_sizes();
    test_partition_rejects_odd_S();
    test_pbo_range();
    test_pbo_from_logit_all_positive();
    test_pbo_from_logit_all_negative();
    test_keep_all_trials();
    std::cout << "All tests passed.\n";
    return 0;
}
