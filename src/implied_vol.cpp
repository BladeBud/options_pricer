#include "implied_vol.hpp"
#include "black_scholes.hpp"
#include <cmath>
#include <algorithm>

namespace pricer {

IVResult iv_newton(double market_price, double S, double K, double T,
                   double r, double q, bool is_call,
                   double tol, int max_iter) {
    IVResult res;
    res.converged = false;
    res.iterations = 0;

    // Initial guess using Brenner-Subrahmanyam approximation
    double sigma = std::sqrt(2.0 * M_PI / T) * market_price / S;
    if (sigma <= 0.0 || !std::isfinite(sigma)) sigma = 0.25;

    for (int i = 0; i < max_iter; ++i) {
        res.iterations = i + 1;

        BSResult g = bs_greeks(S, K, T, r, q, sigma, is_call);
        double diff = g.price - market_price;

        if (std::abs(diff) < tol) {
            res.implied_vol = sigma;
            res.converged = true;
            return res;
        }

        double vega = g.vega;
        if (std::abs(vega) < 1e-20) {
            // Vega too small, can't continue
            break;
        }

        sigma -= diff / vega;

        // Clamp to positive
        if (sigma <= 0.0) sigma = 1e-6;
        if (sigma > 20.0) sigma = 20.0;
    }

    res.implied_vol = sigma;
    return res;
}

IVResult iv_brent(double market_price, double S, double K, double T,
                  double r, double q, bool is_call,
                  double vol_low, double vol_high,
                  double tol, int max_iter) {
    IVResult res;
    res.converged = false;
    res.iterations = 0;

    auto f = [&](double sigma) -> double {
        return bs_price(S, K, T, r, q, sigma, is_call) - market_price;
    };

    double a = vol_low;
    double b = vol_high;
    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0.0) {
        // Root not bracketed; try to find a bracket
        // Expand the search
        for (int i = 0; i < 50; ++i) {
            a *= 0.5;
            fa = f(a);
            if (fa * fb <= 0.0) break;
            b *= 2.0;
            fb = f(b);
            if (fa * fb <= 0.0) break;
        }
        if (fa * fb > 0.0) {
            res.implied_vol = (a + b) * 0.5;
            return res;  // Could not bracket
        }
    }

    double c = a, fc = fa;
    double d = b - a, e = d;

    for (int i = 0; i < max_iter; ++i) {
        res.iterations = i + 1;

        if (fb * fc > 0.0) {
            c = a; fc = fa;
            d = b - a; e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        double tol1 = 2.0 * 2.2204460492503131e-16 * std::abs(b) + 0.5 * tol;
        double m = 0.5 * (c - b);

        if (std::abs(m) <= tol1 || std::abs(fb) < tol) {
            res.implied_vol = b;
            res.converged = true;
            return res;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s_val = fb / fa;
            double p, q_val;
            if (std::abs(a - c) < 1e-15) {
                // Secant
                p = 2.0 * m * s_val;
                q_val = 1.0 - s_val;
            } else {
                // Inverse quadratic interpolation
                double q2 = fa / fc;
                double r2 = fb / fc;
                p = s_val * (2.0 * m * q2 * (q2 - r2) - (b - a) * (r2 - 1.0));
                q_val = (q2 - 1.0) * (r2 - 1.0) * (s_val - 1.0);
            }
            if (p > 0.0) q_val = -q_val;
            else p = -p;

            if (2.0 * p < std::min(3.0 * m * q_val - std::abs(tol1 * q_val), std::abs(e * q_val))) {
                e = d;
                d = p / q_val;
            } else {
                d = m; e = m;
            }
        } else {
            d = m; e = m;
        }

        a = b; fa = fb;
        if (std::abs(d) > tol1)
            b += d;
        else
            b += (m > 0.0 ? tol1 : -tol1);

        fb = f(b);
    }

    res.implied_vol = b;
    return res;
}

} // namespace pricer
