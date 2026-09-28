#pragma once

namespace pricer {

struct IVResult {
    double implied_vol;
    int iterations;
    bool converged;
};

// Newton-Raphson implied volatility solver
// market_price: observed option price, S: spot, K: strike, T: time to expiry,
// r: risk-free rate, q: dividend yield, is_call: call/put
// tol: convergence tolerance, max_iter: maximum iterations
IVResult iv_newton(double market_price, double S, double K, double T,
                   double r, double q, bool is_call,
                   double tol = 1e-10, int max_iter = 100);

// Brent's method implied volatility solver (more robust)
IVResult iv_brent(double market_price, double S, double K, double T,
                  double r, double q, bool is_call,
                  double vol_low = 1e-6, double vol_high = 10.0,
                  double tol = 1e-10, int max_iter = 200);

} // namespace pricer
