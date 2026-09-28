#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "black_scholes.hpp"
#include "binomial.hpp"
#include "monte_carlo.hpp"
#include "implied_vol.hpp"

namespace py = pybind11;

PYBIND11_MODULE(options_engine, m) {
    m.doc() = "High-performance options pricing engine";

    // --- Black-Scholes ---
    py::class_<pricer::BSResult>(m, "BSResult")
        .def_readonly("price", &pricer::BSResult::price)
        .def_readonly("delta", &pricer::BSResult::delta)
        .def_readonly("gamma", &pricer::BSResult::gamma)
        .def_readonly("vega",  &pricer::BSResult::vega)
        .def_readonly("theta", &pricer::BSResult::theta)
        .def_readonly("rho",   &pricer::BSResult::rho)
        .def("__repr__", [](const pricer::BSResult& r) {
            return "<BSResult price=" + std::to_string(r.price)
                 + " delta=" + std::to_string(r.delta)
                 + " gamma=" + std::to_string(r.gamma)
                 + " vega="  + std::to_string(r.vega)
                 + " theta=" + std::to_string(r.theta)
                 + " rho="   + std::to_string(r.rho) + ">";
        });

    m.def("bs_price", &pricer::bs_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("is_call"),
          "Black-Scholes European option price");

    m.def("bs_greeks", &pricer::bs_greeks,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("is_call"),
          "Black-Scholes price and Greeks");

    // --- Binomial ---
    py::class_<pricer::BinomialResult>(m, "BinomialResult")
        .def_readonly("price", &pricer::BinomialResult::price)
        .def_readonly("delta", &pricer::BinomialResult::delta)
        .def_readonly("gamma", &pricer::BinomialResult::gamma)
        .def_readonly("theta", &pricer::BinomialResult::theta)
        .def_readonly("exercise_boundary", &pricer::BinomialResult::exercise_boundary)
        .def("__repr__", [](const pricer::BinomialResult& r) {
            return "<BinomialResult price=" + std::to_string(r.price)
                 + " delta=" + std::to_string(r.delta)
                 + " gamma=" + std::to_string(r.gamma)
                 + " theta=" + std::to_string(r.theta) + ">";
        });

    m.def("crr_price", &pricer::crr_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("steps"), py::arg("is_call"), py::arg("is_american"),
          "CRR binomial tree option price");

    // --- Monte Carlo ---
    py::class_<pricer::MCResult>(m, "MCResult")
        .def_readonly("price", &pricer::MCResult::price)
        .def_readonly("std_error", &pricer::MCResult::std_error)
        .def_readonly("ci_lower", &pricer::MCResult::ci_lower)
        .def_readonly("ci_upper", &pricer::MCResult::ci_upper)
        .def("__repr__", [](const pricer::MCResult& r) {
            return "<MCResult price=" + std::to_string(r.price)
                 + " stderr=" + std::to_string(r.std_error)
                 + " CI=[" + std::to_string(r.ci_lower)
                 + ", " + std::to_string(r.ci_upper) + "]>";
        });

    m.def("mc_price", &pricer::mc_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("num_paths"), py::arg("seed"), py::arg("is_call"),
          "Monte Carlo European option price (single-step, antithetic)");

    m.def("mc_price_paths", &pricer::mc_price_paths,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("num_paths"), py::arg("num_steps"),
          py::arg("seed"), py::arg("is_call"),
          "Monte Carlo European option price (multi-step, antithetic)");

    // --- Implied Volatility ---
    py::class_<pricer::IVResult>(m, "IVResult")
        .def_readonly("implied_vol", &pricer::IVResult::implied_vol)
        .def_readonly("iterations", &pricer::IVResult::iterations)
        .def_readonly("converged", &pricer::IVResult::converged)
        .def("__repr__", [](const pricer::IVResult& r) {
            return "<IVResult iv=" + std::to_string(r.implied_vol)
                 + " iters=" + std::to_string(r.iterations)
                 + " converged=" + std::to_string(r.converged) + ">";
        });

    m.def("iv_newton", &pricer::iv_newton,
          py::arg("market_price"), py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("is_call"),
          py::arg("tol") = 1e-10, py::arg("max_iter") = 100,
          "Newton-Raphson implied volatility solver");

    m.def("iv_brent", &pricer::iv_brent,
          py::arg("market_price"), py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("is_call"),
          py::arg("vol_low") = 1e-6, py::arg("vol_high") = 10.0,
          py::arg("tol") = 1e-10, py::arg("max_iter") = 200,
          "Brent's method implied volatility solver");
}
