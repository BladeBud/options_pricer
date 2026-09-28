#pragma once
#include <cmath>
#include <stdexcept>

namespace pricer {

struct BSResult {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// Standard normal PDF
inline double norm_pdf(double x) noexcept {
    constexpr double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// Standard normal CDF
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Black-Scholes closed-form price for European options
// is_call: true for call, false for put
// S: spot, K: strike, T: time to expiry (years), r: risk-free rate, q: dividend yield, sigma: volatility
double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call);

// Full Greeks computation
BSResult bs_greeks(double S, double K, double T, double r, double q, double sigma, bool is_call);

} // namespace pricer
