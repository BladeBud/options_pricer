#include "black_scholes.hpp"
#include <algorithm>

namespace pricer {

double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call) {
    if (T <= 0.0) {
        // At expiry
        double intrinsic = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return intrinsic;
    }
    if (sigma <= 0.0) {
        // Zero vol: deterministic
        double forward = S * std::exp((r - q) * T);
        double df = std::exp(-r * T);
        double intrinsic = is_call ? std::max(forward - K, 0.0) : std::max(K - forward, 0.0);
        return df * intrinsic;
    }

    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);

    if (is_call) {
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
    } else {
        return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
    }
}

BSResult bs_greeks(double S, double K, double T, double r, double q, double sigma, bool is_call) {
    BSResult res{};

    if (T <= 0.0 || sigma <= 0.0) {
        res.price = bs_price(S, K, T, r, q, sigma, is_call);
        // Greeks at expiry or zero vol are edge-case; return zeros
        return res;
    }

    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);
    double nd1 = norm_cdf(d1);
    double nd2 = norm_cdf(d2);
    double pd1 = norm_pdf(d1);

    if (is_call) {
        res.price = S * df_q * nd1 - K * df_r * nd2;
        res.delta = df_q * nd1;
        res.theta = (-S * df_q * pd1 * sigma / (2.0 * sqrt_T)
                     + q * S * df_q * nd1
                     - r * K * df_r * nd2);
        res.rho = K * T * df_r * nd2;
    } else {
        double nmd1 = norm_cdf(-d1);
        double nmd2 = norm_cdf(-d2);
        res.price = K * df_r * nmd2 - S * df_q * nmd1;
        res.delta = -df_q * nmd1;
        res.theta = (-S * df_q * pd1 * sigma / (2.0 * sqrt_T)
                     - q * S * df_q * nmd1
                     + r * K * df_r * nmd2);
        res.rho = -K * T * df_r * nmd2;
    }

    // Gamma and Vega are the same for calls and puts
    res.gamma = df_q * pd1 / (S * sigma * sqrt_T);
    res.vega  = S * df_q * pd1 * sqrt_T;

    return res;
}

} // namespace pricer
