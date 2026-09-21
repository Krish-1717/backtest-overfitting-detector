/**
 * src/cscv.cpp -- CSCV implementation
 * Day 9 Commit 2
 */
#include "cscv.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace backtest {

static double _mean(const double* v, std::size_t n) {
    double s = 0; for (std::size_t i=0;i<n;++i) s+=v[i]; return s/n;
}
static double _std(const double* v, std::size_t n, double m) {
    double s = 0; for (std::size_t i=0;i<n;++i) s+=(v[i]-m)*(v[i]-m);
    return n>1 ? std::sqrt(s/(n-1)) : 0.0;
}

double cscv_sharpe(const double* rets, std::size_t n) {
    if (!n) return std::numeric_limits<double>::quiet_NaN();
    double m = _mean(rets, n), s = _std(rets, n, m);
    return s < 1e-12 ? std::numeric_limits<double>::quiet_NaN() : m/s*std::sqrt(252.0);
}

static std::vector<double> col(const Matrix& m, std::size_t c) {
    std::vector<double> v(m.rows);
    for (std::size_t r=0;r<m.rows;++r) v[r]=m.at(r,c);
    return v;
}

static Matrix merge_parts(const std::vector<Matrix>& parts,
                           const std::vector<std::size_t>& idx) {
    std::size_t N=parts[0].cols, T=0;
    for (auto i:idx) T+=parts[i].rows;
    Matrix out; out.rows=T; out.cols=N; out.data.resize(T*N);
    std::size_t row=0;
    for (auto i:idx)
        for (std::size_t r=0;r<parts[i].rows;++r,++row)
            for (std::size_t c=0;c<N;++c)
                out.at(row,c)=parts[i].at(r,c);
    return out;
}

std::vector<Matrix> cscv_partition(const Matrix& returns, std::size_t S) {
    if (S<2||S%2!=0) throw std::invalid_argument("S must be even >= 2");
    if (returns.rows<S) throw std::invalid_argument("Too few rows for S partitions");
    std::size_t base=returns.rows/S, extra=returns.rows%S;
    std::vector<Matrix> slices; slices.reserve(S);
    std::size_t row=0;
    for (std::size_t s=0;s<S;++s) {
        std::size_t len=base+(s<extra?1:0);
        Matrix sl; sl.rows=len; sl.cols=returns.cols; sl.data.resize(len*returns.cols);
        for (std::size_t r=0;r<len;++r,++row)
            for (std::size_t c=0;c<returns.cols;++c)
                sl.at(r,c)=returns.at(row,c);
        slices.push_back(std::move(sl));
    }
    return slices;
}

double pbo_from_logit(const std::vector<double>& lam) {
    if (lam.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::size_t neg=0;
    for (double v:lam) if (std::isfinite(v)&&v<0.0) ++neg;
    return static_cast<double>(neg)/lam.size();
}

static void combinations(std::size_t S, std::size_t half,
                          std::vector<std::vector<std::size_t>>& out) {
    std::vector<std::size_t> c;
    std::function<void(std::size_t,std::size_t)> gen=[&](std::size_t st,std::size_t rem){
        if (!rem){out.push_back(c);return;}
        for (std::size_t i=st;i+rem<=S;++i){c.push_back(i);gen(i+1,rem-1);c.pop_back();}
    };
    gen(0,half);
}

CscvResult cscv_pbo(const Matrix& returns, std::size_t S, bool keep_all) {
    if (S<4||S%2!=0) throw std::invalid_argument("S must be even >= 4");
    std::size_t N=returns.cols;
    if (N<2) throw std::invalid_argument("Need >= 2 strategies");
    auto parts=cscv_partition(returns,S);
    std::size_t half=S/2;
    std::vector<std::vector<std::size_t>> is_sets;
    combinations(S,half,is_sets);
    CscvResult res; res.n_trials=is_sets.size();
    if (keep_all) res.trials.reserve(res.n_trials);
    std::vector<double> lam; lam.reserve(res.n_trials);
    double oos_sum=0; std::size_t valid=0;
    for (auto& is_idx:is_sets) {
        std::vector<std::size_t> oos_idx;
        for (std::size_t s=0;s<S;++s)
            if (std::find(is_idx.begin(),is_idx.end(),s)==is_idx.end())
                oos_idx.push_back(s);
        auto is_mat=merge_parts(parts,is_idx), oos_mat=merge_parts(parts,oos_idx);
        std::vector<double> is_sr(N),oos_sr(N);
        for (std::size_t c=0;c<N;++c){
            auto ic=col(is_mat,c),oc=col(oos_mat,c);
            is_sr[c]=cscv_sharpe(ic.data(),ic.size());
            oos_sr[c]=cscv_sharpe(oc.data(),oc.size());
        }
        int best=static_cast<int>(std::max_element(is_sr.begin(),is_sr.end())-is_sr.begin());
        double oos_best=oos_sr[best];
        auto sorted_oos=oos_sr; std::sort(sorted_oos.begin(),sorted_oos.end());
        std::size_t rank=static_cast<std::size_t>(
            std::lower_bound(sorted_oos.begin(),sorted_oos.end(),oos_best)-sorted_oos.begin())+1;
        double rel=std::max(1e-9,std::min(1.0-1e-9,(double)rank/(N+1)));
        double lc=std::log(rel/(1.0-rel));
        lam.push_back(lc);
        if (std::isfinite(oos_best)){oos_sum+=oos_best;++valid;}
        if (keep_all) res.trials.push_back({best,is_sr[best],oos_best,lc});
    }
    res.pbo=pbo_from_logit(lam);
    res.expected_oos_sharpe=valid?oos_sum/valid:0.0;
    std::size_t def=0; for (double v:lam) if (std::isfinite(v)&&v<0) ++def;
    res.rank_deficiency=static_cast<double>(def)/lam.size();
    return res;
}
} // namespace backtest
