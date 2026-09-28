#include "monte_carlo.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace pricer {

MCResult mc_price(double S, double K, double T, double r, double q,
                  double sigma, uint64_t num_paths, uint64_t seed, bool is_call) {
    // Single-step GBM with antithetic variates
    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;
    uint64_t total_paths = num_paths;  // each iteration produces 2 paths (antithetic)

#ifdef USE_OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    // Per-thread accumulators
    std::vector<double> thread_sum(num_threads, 0.0);
    std::vector<double> thread_sum_sq(num_threads, 0.0);

#ifdef USE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef USE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937_64 rng(seed + static_cast<uint64_t>(tid) * 1000003ULL);
        std::normal_distribution<double> norm(0.0, 1.0);

        double local_sum = 0.0;
        double local_sum_sq = 0.0;

        // Each iteration generates a pair (antithetic)
        uint64_t half = num_paths / 2;

#ifdef USE_OPENMP
        #pragma omp for schedule(static)
#endif
        for (uint64_t i = 0; i < half; ++i) {
            double z = norm(rng);

            // Path 1
            double S1 = S * std::exp(drift + vol_sqrt_T * z);
            double payoff1 = is_call ? std::max(S1 - K, 0.0) : std::max(K - S1, 0.0);

            // Antithetic path
            double S2 = S * std::exp(drift - vol_sqrt_T * z);
            double payoff2 = is_call ? std::max(S2 - K, 0.0) : std::max(K - S2, 0.0);

            // Average of the pair for variance reduction
            double avg_payoff = 0.5 * (payoff1 + payoff2);
            local_sum += avg_payoff;
            local_sum_sq += avg_payoff * avg_payoff;
        }

        thread_sum[tid] = local_sum;
        thread_sum_sq[tid] = local_sum_sq;
    }

    for (int t = 0; t < num_threads; ++t) {
        sum_payoff += thread_sum[t];
        sum_payoff_sq += thread_sum_sq[t];
    }

    uint64_t half = num_paths / 2;
    double mean = sum_payoff / static_cast<double>(half);
    double variance = (sum_payoff_sq / static_cast<double>(half)) - mean * mean;
    double std_err = std::sqrt(std::max(variance, 0.0) / static_cast<double>(half));

    MCResult res;
    res.price = disc * mean;
    res.std_error = disc * std_err;
    res.ci_lower = res.price - 1.96 * res.std_error;
    res.ci_upper = res.price + 1.96 * res.std_error;
    return res;
}

MCResult mc_price_paths(double S, double K, double T, double r, double q,
                        double sigma, uint64_t num_paths, int num_steps,
                        uint64_t seed, bool is_call) {
    // Multi-step GBM with antithetic variates
    double dt = T / num_steps;
    double drift_dt = (r - q - 0.5 * sigma * sigma) * dt;
    double vol_sqrt_dt = sigma * std::sqrt(dt);
    double disc = std::exp(-r * T);

#ifdef USE_OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    std::vector<double> thread_sum(num_threads, 0.0);
    std::vector<double> thread_sum_sq(num_threads, 0.0);

    uint64_t half = num_paths / 2;

#ifdef USE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef USE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937_64 rng(seed + static_cast<uint64_t>(tid) * 1000003ULL);
        std::normal_distribution<double> norm(0.0, 1.0);

        double local_sum = 0.0;
        double local_sum_sq = 0.0;

#ifdef USE_OPENMP
        #pragma omp for schedule(static)
#endif
        for (uint64_t i = 0; i < half; ++i) {
            double S1 = S;
            double S2 = S;

            for (int step = 0; step < num_steps; ++step) {
                double z = norm(rng);
                S1 *= std::exp(drift_dt + vol_sqrt_dt * z);
                S2 *= std::exp(drift_dt - vol_sqrt_dt * z);
            }

            double payoff1 = is_call ? std::max(S1 - K, 0.0) : std::max(K - S1, 0.0);
            double payoff2 = is_call ? std::max(S2 - K, 0.0) : std::max(K - S2, 0.0);
            double avg = 0.5 * (payoff1 + payoff2);

            local_sum += avg;
            local_sum_sq += avg * avg;
        }

        thread_sum[tid] = local_sum;
        thread_sum_sq[tid] = local_sum_sq;
    }

    double sum = 0.0, sum_sq = 0.0;
    for (int t = 0; t < num_threads; ++t) {
        sum += thread_sum[t];
        sum_sq += thread_sum_sq[t];
    }

    double mean = sum / static_cast<double>(half);
    double var = (sum_sq / static_cast<double>(half)) - mean * mean;
    double se = std::sqrt(std::max(var, 0.0) / static_cast<double>(half));

    MCResult res;
    res.price = disc * mean;
    res.std_error = disc * se;
    res.ci_lower = res.price - 1.96 * res.std_error;
    res.ci_upper = res.price + 1.96 * res.std_error;
    return res;
}

} // namespace pricer
