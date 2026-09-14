/**
 * test_stats.cpp -- Unit tests for stats and PBO modules (37 tests)
 *
 * No external framework -- standalone binary.
 * Build: cmake .. && make && ctest --output-on-failure
 * Or:    ./test_stats
 * Expected final line: "ALL TESTS PASSED"
 */

#include "../src/stats.h"
#include "../src/pbo.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

static int g_pass = 0, g_fail = 0;
#define CHECK(name, cond) do { if (cond) { ++g_pass; std::cout << "  [OK] " << name << "\n"; } else { ++g_fail; std::cout << "  [FAIL] " << name << "\n"; } } while(0)
#define CHECK_NEAR(name, a, b, tol) CHECK(name, std::fabs((a)-(b)) < (tol))

static std::vector<double> make_returns(int n, double drift, double vol, int seed=0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(drift, vol);
    std::vector<double> r(n);
    for (auto& x : r) x = d(rng);
    return r;
}

void test_moments() {
    std::cout << "\n[Moments]\n";
    std::vector<double> v = {1,2,3,4,5};
    CHECK_NEAR("mean", detector::mean(v), 3.0, 1e-10);
    CHECK_NEAR("variance", detector::variance(v), 2.5, 1e-10);
    CHECK_NEAR("std_dev", detector::std_dev(v), std::sqrt(2.5), 1e-10);
    auto sym = make_returns(5000, 0.0, 1.0, 1);
    CHECK_NEAR("skewness N(0,1) ~= 0", detector::skewness(sym), 0.0, 0.15);
    CHECK_NEAR("excess kurtosis N(0,1) ~= 0", detector::excess_kurtosis(sym), 0.0, 0.2);
    CHECK("mean of empty = 0", detector::mean({}) == 0.0);
    CHECK("variance of 1 elem = 0", detector::variance({5.0}) == 0.0);
}

void test_sharpe() {
    std::cout << "\n[Sharpe ratio]\n";
    auto noise = make_returns(5000, 0.0, 0.01, 42);
    CHECK("noise SR near 0", std::fabs(detector::sharpe_ratio(noise)) < 0.5);
    auto signal = make_returns(10000, 1.0*0.01/std::sqrt(252.0), 0.01, 42);
    CHECK_NEAR("signal SR ~= 1.0", detector::sharpe_ratio(signal), 1.0, 0.25);
    CHECK("SR of single = 0", detector::sharpe_ratio({0.01}) == 0.0);
}

void test_normal_cdf() {
    std::cout << "\n[Normal CDF]\n";
    CHECK_NEAR("Phi(0) = 0.5", detector::normal_cdf(0.0), 0.5, 1e-6);
    CHECK_NEAR("Phi(1.96) ~= 0.975", detector::normal_cdf(1.96), 0.9750, 2e-4);
    CHECK_NEAR("Phi(-1.96) ~= 0.025", detector::normal_cdf(-1.96), 0.0250, 2e-4);
    CHECK_NEAR("Phi(9) = 1", detector::normal_cdf(9.0), 1.0, 1e-6);
    CHECK_NEAR("Phi(-9) = 0", detector::normal_cdf(-9.0), 0.0, 1e-6);
    for (double p : {0.01, 0.1, 0.5, 0.9, 0.99}) {
        double x = detector::normal_icdf(p);
        CHECK_NEAR("CDF round-trip p="+std::to_string(p), detector::normal_cdf(x), p, 1e-5);
    }
}

void test_psr() {
    std::cout << "\n[PSR]\n";
    auto noise = make_returns(5000, 0.0, 0.01, 1);
    CHECK_NEAR("PSR of noise vs SR*=0 ~= 0.5", detector::psr(noise, 0.0), 0.5, 0.15);
    auto strong = make_returns(5000, 2.0*0.01/std::sqrt(252.0), 0.01, 2);
    CHECK("PSR of SR~2 strategy > 0.9", detector::psr(strong, 0.5) > 0.9);
    for (int s = 0; s < 5; ++s) {
        auto r = make_returns(252, 0.0, 0.01, s);
        double p = detector::psr(r, 0.5);
        CHECK("PSR in [0,1] seed="+std::to_string(s), p >= 0.0 && p <= 1.0);
    }
}

void test_dsr() {
    std::cout << "\n[DSR]\n";
    auto noise = make_returns(252, 0.0, 0.01, 7);
    double psr_v = detector::psr(noise, 0.5);
    double dsr_v = detector::dsr(noise, 100, 0.5);
    CHECK("DSR <= PSR", dsr_v <= psr_v);
    CHECK("DSR in [0,1]", dsr_v >= 0.0 && dsr_v <= 1.0);
    double e1 = detector::expected_max_sharpe(10, 0.5, 252);
    double e2 = detector::expected_max_sharpe(100, 0.5, 252);
    double e3 = detector::expected_max_sharpe(1000, 0.5, 252);
    CHECK("E[max SR] increases with N", e1 < e2 && e2 < e3);
}

void test_cscv() {
    std::cout << "\n[CSCV / PBO]\n";
    auto combos = detector::combinations(4, 2);
    CHECK("C(4,2) = 6", combos.size() == 6);

    std::vector<std::vector<double>> noise_strats;
    for (int s = 0; s < 200; ++s) noise_strats.push_back(make_returns(504, 0.0, 0.01, s));
    auto res_noise = detector::run_cscv(noise_strats, 8, 252.0, 200);
    CHECK("PBO of noise >= 0.4", res_noise.pbo >= 0.4);
    CHECK("PBO in [0,1]", res_noise.pbo >= 0.0 && res_noise.pbo <= 1.0);
    CHECK("CSCV n_trials > 0", res_noise.n_trials > 0);

    std::vector<std::vector<double>> mixed;
    for (int s = 0; s < 200; ++s) {
        double d = (s < 40) ? (2.0*0.01/std::sqrt(252.0)) : 0.0;
        mixed.push_back(make_returns(504, d, 0.01, s + 500));
    }
    auto res_mix = detector::run_cscv(mixed, 8, 252.0, 200);
    CHECK("mixed PBO <= noise PBO + 0.1", res_mix.pbo <= res_noise.pbo + 0.1);
    CHECK_NEAR("noise avg OOS Sharpe ~= 0", res_noise.avg_oos_sharpe, 0.0, 0.8);
}

void test_noise_detection() {
    std::cout << "\n[Noise detection]\n";
    std::mt19937 rng(99);
    std::vector<std::vector<double>> all_noise, mixed;
    for (int s = 0; s < 100; ++s) {
        std::normal_distribution<double> d(0.0, 0.01);
        std::vector<double> r(252);
        for (auto& x : r) x = d(rng);
        all_noise.push_back(r);
    }
    for (int s = 0; s < 100; ++s) {
        double drift = (s < 10) ? (1.0*0.01/std::sqrt(252.0)) : 0.0;
        std::normal_distribution<double> d(drift, 0.01);
        std::vector<double> r(252);
        for (auto& x : r) x = d(rng);
        mixed.push_back(r);
    }
    auto pbo_n = detector::run_cscv(all_noise, 8, 252.0, 200);
    auto pbo_m = detector::run_cscv(mixed, 8, 252.0, 200);
    std::cout << "  all-noise PBO=" << pbo_n.pbo << "  mixed PBO=" << pbo_m.pbo << "\n";
    CHECK("all-noise PBO >= mixed PBO - 0.1", pbo_n.pbo >= pbo_m.pbo - 0.1);
}

int main() {
    std::cout << "=== Backtest Overfitting Detector -- Unit Tests ===\n";
    test_moments();
    test_sharpe();
    test_normal_cdf();
    test_psr();
    test_dsr();
    test_cscv();
    test_noise_detection();
    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    if (g_fail == 0) { std::cout << "\nALL TESTS PASSED\n"; return 0; }
    std::cout << "\n" << g_fail << " TESTS FAILED\n"; return 1;
}
