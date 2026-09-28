#pragma once
#include <cstdint>

namespace pricer {

struct MCResult {
    double price;
    double std_error;
    double ci_lower;  // 95% confidence interval
    double ci_upper;
};

// Monte Carlo European option pricer using GBM with antithetic variates
// S: spot, K: strike, T: time to expiry, r: risk-free rate, q: dividend yield,
// sigma: volatility, num_paths: number of simulation paths, seed: RNG seed,
// is_call: call/put
MCResult mc_price(double S, double K, double T, double r, double q,
                  double sigma, uint64_t num_paths, uint64_t seed, bool is_call);

// Monte Carlo with path-dependent pricing (for future extensibility)
// num_steps: number of time steps per path
MCResult mc_price_paths(double S, double K, double T, double r, double q,
                        double sigma, uint64_t num_paths, int num_steps,
                        uint64_t seed, bool is_call);

} // namespace pricer
