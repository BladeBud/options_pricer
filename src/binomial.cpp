#include "binomial.hpp"
#include <cmath>
#include <algorithm>

namespace pricer {

BinomialResult crr_price(double S, double K, double T, double r, double q,
                         double sigma, int steps, bool is_call, bool is_american) {
    BinomialResult result{};
    if (steps <= 0) steps = 1;

    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double disc = std::exp(-r * dt);
    double drift = std::exp((r - q) * dt);
    double p_up = (drift - d) / (u - d);
    double p_dn = 1.0 - p_up;

    int n = steps + 1;

    // Use a single vector, overwriting in place for O(n) memory
    std::vector<double> prices(n);
    std::vector<double> option(n);

    // Terminal stock prices and payoffs
    double S_base = S * std::pow(d, steps);
    for (int i = 0; i < n; ++i) {
        prices[i] = S_base * std::pow(u / d, i);
        double payoff = is_call ? std::max(prices[i] - K, 0.0) : std::max(K - prices[i], 0.0);
        option[i] = payoff;
    }

    // Backward induction
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            prices[i] = S * std::pow(u, 2 * i - step);  // recalculate to avoid drift
            option[i] = disc * (p_up * option[i + 1] + p_dn * option[i]);

            if (is_american) {
                double intrinsic = is_call ? std::max(prices[i] - K, 0.0) : std::max(K - prices[i], 0.0);
                if (intrinsic > option[i]) {
                    option[i] = intrinsic;
                    // Record exercise boundary point
                    result.exercise_boundary.emplace_back(step, prices[i]);
                }
            }
        }

        // Extract Greeks at step 2 and step 1
        if (step == 2 && steps >= 2) {
            double S_uu = S * u * u;
            double S_ud = S;  // u*d = 1
            double S_dd = S * d * d;
            double f_uu = option[2];
            double f_ud = option[1];
            double f_dd = option[0];
            result.gamma = ((f_uu - f_ud) / (S_uu - S_ud) - (f_ud - f_dd) / (S_ud - S_dd))
                           / (0.5 * (S_uu - S_dd));
            result.theta = (f_ud - option[1]) / (2.0 * dt);  // Will be corrected below
        }
        if (step == 1) {
            double S_u = S * u;
            double S_d = S * d;
            result.delta = (option[1] - option[0]) / (S_u - S_d);
        }
    }

    result.price = option[0];

    // Theta correction: need to compare f(0,0) with f(1, mid)
    // Using the standard approach: theta = (f_ud - f_00) / (2*dt) where f_ud is at step 2 center
    // This was already handled above if steps >= 2
    if (steps >= 2) {
        // Recompute to get theta properly
        // We already stored it during backward induction but let's fix it
        // theta = (option_value_at_step2_middle - option_value_at_step0) / (2 * dt)
        // We need to rerun to get the step2 middle value... let's use a simpler approach:
        // Just recompute for step 2
        std::vector<double> opt2(steps + 1);
        S_base = S * std::pow(d, steps);
        for (int i = 0; i < n; ++i) {
            double Si = S_base * std::pow(u / d, i);
            opt2[i] = is_call ? std::max(Si - K, 0.0) : std::max(K - Si, 0.0);
        }
        for (int step2 = steps - 1; step2 >= 2; --step2) {
            for (int i = 0; i <= step2; ++i) {
                double Si = S * std::pow(u, 2 * i - step2);
                opt2[i] = disc * (p_up * opt2[i + 1] + p_dn * opt2[i]);
                if (is_american) {
                    double intrinsic = is_call ? std::max(Si - K, 0.0) : std::max(K - Si, 0.0);
                    opt2[i] = std::max(opt2[i], intrinsic);
                }
            }
        }
        // opt2[1] is the center node at step=2 (the node corresponding to one up, one down = S)
        result.theta = (opt2[1] - result.price) / (2.0 * dt);
    }

    return result;
}

} // namespace pricer
