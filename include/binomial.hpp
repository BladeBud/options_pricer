#pragma once
#include <vector>

namespace pricer {

struct BinomialResult {
    double price;
    double delta;
    double gamma;
    double theta;
    // Early exercise boundary: vector of (time_step, critical_stock_price)
    std::vector<std::pair<int, double>> exercise_boundary;
};

// Cox-Ross-Rubinstein binomial tree pricer
// Supports American and European options
// S: spot, K: strike, T: time to expiry, r: risk-free rate, q: dividend yield,
// sigma: volatility, steps: number of time steps, is_call: call/put, is_american: American/European
BinomialResult crr_price(double S, double K, double T, double r, double q,
                         double sigma, int steps, bool is_call, bool is_american);

} // namespace pricer
