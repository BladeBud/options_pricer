#!/usr/bin/env python3
"""Options Pricing & Volatility Surface Calibrator.

Fetches a live options chain via yfinance, computes implied volatilities
using the C++ engine, and plots a 3D volatility surface.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Ensure the build directory is on the path so we can import options_engine
_build_dir = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(_build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import options_engine as engine
except ImportError:
    print("ERROR: Cannot import options_engine.")
    print(f"Looked in: {_build_dir}")
    print("Make sure you have built the C++ module:")
    print("  cd options-pricer && mkdir -p build && cd build && cmake .. && make -j")
    sys.exit(1)


def fetch_options_data(ticker: str = "SPY") -> pd.DataFrame:
    """Fetch the full options chain for a ticker."""
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    expirations = tk.options

    if not expirations:
        print(f"No options data available for {ticker}")
        sys.exit(1)

    print(f"Ticker: {ticker}")
    print(f"Spot price: ${spot:.2f}")
    print(f"Expirations available: {len(expirations)}")

    rows = []
    now = datetime.now()

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        T = (exp_date - now).days / 365.25
        if T <= 0.001:
            continue  # skip expired

        chain = tk.option_chain(exp_str)

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                mid = (row.get("bid", 0.0) + row.get("ask", 0.0)) / 2.0
                if mid <= 0.01:
                    continue  # no meaningful price
                volume = row.get("volume", 0)
                if volume is None or (isinstance(volume, float) and np.isnan(volume)):
                    volume = 0
                oi = row.get("openInterest", 0)
                if oi is None or (isinstance(oi, float) and np.isnan(oi)):
                    oi = 0

                rows.append({
                    "expiration": exp_str,
                    "T": T,
                    "strike": row["strike"],
                    "mid_price": mid,
                    "bid": row.get("bid", 0.0),
                    "ask": row.get("ask", 0.0),
                    "volume": int(volume),
                    "open_interest": int(oi),
                    "type": opt_type,
                    "spot": spot,
                })

    df = pd.DataFrame(rows)
    print(f"Total option contracts fetched: {len(df)}")
    return df


def compute_implied_vols(df: pd.DataFrame, r: float = 0.05, q: float = 0.013) -> pd.DataFrame:
    """Compute implied volatility for each option using the C++ engine."""
    ivs = []
    for _, row in df.iterrows():
        is_call = row["type"] == "call"
        try:
            result = engine.iv_brent(
                market_price=row["mid_price"],
                S=row["spot"],
                K=row["strike"],
                T=row["T"],
                r=r,
                q=q,
                is_call=is_call,
            )
            if result.converged and 0.001 < result.implied_vol < 5.0:
                ivs.append(result.implied_vol)
            else:
                ivs.append(np.nan)
        except Exception:
            ivs.append(np.nan)

    df = df.copy()
    df["implied_vol"] = ivs
    before = len(df)
    df = df.dropna(subset=["implied_vol"])
    print(f"IV computed: {len(df)}/{before} contracts converged")
    return df


def compute_greeks(df: pd.DataFrame, r: float = 0.05, q: float = 0.013) -> pd.DataFrame:
    """Compute Black-Scholes Greeks using computed IV."""
    greeks_data = []
    for _, row in df.iterrows():
        is_call = row["type"] == "call"
        g = engine.bs_greeks(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=r,
            q=q,
            sigma=row["implied_vol"],
            is_call=is_call,
        )
        greeks_data.append({
            "bs_price": g.price,
            "delta": g.delta,
            "gamma": g.gamma,
            "vega": g.vega,
            "theta": g.theta,
            "rho": g.rho,
        })

    greeks_df = pd.DataFrame(greeks_data, index=df.index)
    return pd.concat([df, greeks_df], axis=1)


def plot_vol_surface(df: pd.DataFrame, ticker: str, option_type: str = "call"):
    """Plot a 3D implied volatility surface."""
    subset = df[df["type"] == option_type].copy()
    if len(subset) < 10:
        print(f"Not enough {option_type} data points for surface plot ({len(subset)})")
        return

    # Filter to reasonable moneyness range
    spot = subset["spot"].iloc[0]
    subset = subset[(subset["strike"] > spot * 0.7) & (subset["strike"] < spot * 1.3)]

    strikes = subset["strike"].values
    expiries = subset["T"].values
    ivs = subset["implied_vol"].values * 100  # Convert to percentage

    # Create a regular grid for interpolation
    strike_grid = np.linspace(strikes.min(), strikes.max(), 60)
    expiry_grid = np.linspace(expiries.min(), expiries.max(), 40)
    X, Y = np.meshgrid(strike_grid, expiry_grid)

    try:
        Z = griddata(
            (strikes, expiries), ivs, (X, Y),
            method="cubic", fill_value=np.nan
        )
    except Exception:
        Z = griddata(
            (strikes, expiries), ivs, (X, Y),
            method="linear", fill_value=np.nan
        )

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)"),
        opacity=0.9,
    )])

    fig.update_layout(
        title=f"{ticker} Implied Volatility Surface ({option_type.title()}s)",
        scene=dict(
            xaxis_title="Strike ($)",
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility (%)",
        ),
        width=1000,
        height=700,
    )

    outfile = Path(__file__).resolve().parent / f"{ticker.lower()}_vol_surface_{option_type}.html"
    fig.write_html(str(outfile))
    print(f"Surface plot saved: {outfile}")

    # Also save as static image if kaleido is available
    try:
        png_file = outfile.with_suffix(".png")
        fig.write_image(str(png_file), scale=2)
        print(f"PNG saved: {png_file}")
    except Exception:
        pass  # kaleido not installed


def plot_smile(df: pd.DataFrame, ticker: str):
    """Plot volatility smile for the nearest expiration."""
    nearest_exp = df.loc[df["T"].idxmin(), "expiration"]
    subset = df[df["expiration"] == nearest_exp]
    spot = subset["spot"].iloc[0]

    fig = go.Figure()

    for opt_type in ["call", "put"]:
        data = subset[subset["type"] == opt_type].sort_values("strike")
        if len(data) < 3:
            continue
        # Filter to reasonable moneyness
        data = data[(data["strike"] > spot * 0.8) & (data["strike"] < spot * 1.2)]
        fig.add_trace(go.Scatter(
            x=data["strike"],
            y=data["implied_vol"] * 100,
            mode="lines+markers",
            name=f"{opt_type.title()}s",
        ))

    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot=${spot:.2f}")

    fig.update_layout(
        title=f"{ticker} Volatility Smile — Expiry: {nearest_exp}",
        xaxis_title="Strike ($)",
        yaxis_title="Implied Volatility (%)",
        width=900,
        height=500,
    )

    outfile = Path(__file__).resolve().parent / f"{ticker.lower()}_vol_smile.html"
    fig.write_html(str(outfile))
    print(f"Smile plot saved: {outfile}")


def run_demo_pricing(spot: float):
    """Run a quick demo of all pricing engines."""
    print("\n" + "=" * 60)
    print("ENGINE DEMO")
    print("=" * 60)

    K = round(spot)
    T = 0.25
    r = 0.05
    q = 0.013
    sigma = 0.20

    print(f"\nParameters: S={spot:.2f}, K={K}, T={T}, r={r}, q={q}, σ={sigma}")

    # Black-Scholes
    bs = engine.bs_greeks(spot, K, T, r, q, sigma, True)
    print(f"\n[Black-Scholes Call]")
    print(f"  Price:  {bs.price:.4f}")
    print(f"  Delta:  {bs.delta:.4f}")
    print(f"  Gamma:  {bs.gamma:.4f}")
    print(f"  Vega:   {bs.vega:.4f}")
    print(f"  Theta:  {bs.theta:.4f}")
    print(f"  Rho:    {bs.rho:.4f}")

    # Binomial
    crr = engine.crr_price(spot, K, T, r, q, sigma, 500, True, True)
    print(f"\n[CRR Binomial American Call — 500 steps]")
    print(f"  Price:  {crr.price:.4f}")
    print(f"  Delta:  {crr.delta:.4f}")

    crr_put = engine.crr_price(spot, K, T, r, q, sigma, 500, False, True)
    crr_eu_put = engine.crr_price(spot, K, T, r, q, sigma, 500, False, False)
    print(f"\n[CRR American Put vs European Put — 500 steps]")
    print(f"  American Put: {crr_put.price:.4f}")
    print(f"  European Put: {crr_eu_put.price:.4f}")
    print(f"  Early exercise premium: {crr_put.price - crr_eu_put.price:.4f}")

    # Monte Carlo
    mc = engine.mc_price(spot, K, T, r, q, sigma, 1_000_000, 42, True)
    print(f"\n[Monte Carlo Call — 1M paths, antithetic]")
    print(f"  Price:  {mc.price:.4f}")
    print(f"  StdErr: {mc.std_error:.6f}")
    print(f"  95% CI: [{mc.ci_lower:.4f}, {mc.ci_upper:.4f}]")

    # Implied Volatility round-trip
    iv = engine.iv_newton(bs.price, spot, K, T, r, q, True)
    print(f"\n[IV Round-trip (Newton)]")
    print(f"  Input σ:    {sigma:.6f}")
    print(f"  Recovered:  {iv.implied_vol:.6f}")
    print(f"  Iterations: {iv.iterations}")
    print(f"  Converged:  {iv.converged}")

    iv2 = engine.iv_brent(bs.price, spot, K, T, r, q, True)
    print(f"\n[IV Round-trip (Brent)]")
    print(f"  Input σ:    {sigma:.6f}")
    print(f"  Recovered:  {iv2.implied_vol:.6f}")
    print(f"  Iterations: {iv2.iterations}")
    print(f"  Converged:  {iv2.converged}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Options Volatility Surface Calibrator")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    parser.add_argument("--div-yield", type=float, default=0.013, help="Dividend yield (default: 0.013)")
    parser.add_argument("--demo-only", action="store_true", help="Only run engine demo, skip live data")
    args = parser.parse_args()

    if args.demo_only:
        run_demo_pricing(450.0)  # Approximate SPY
        return

    # Fetch live data
    print("Fetching options chain...")
    df = fetch_options_data(args.ticker)

    if df.empty:
        print("No data fetched.")
        return

    spot = df["spot"].iloc[0]

    # Run engine demo with live spot
    run_demo_pricing(spot)

    # Compute implied volatilities
    print("\n" + "=" * 60)
    print("CALIBRATING IMPLIED VOLATILITIES")
    print("=" * 60)
    df = compute_implied_vols(df, r=args.rate, q=args.div_yield)

    if df.empty:
        print("No valid IV data.")
        return

    # Compute Greeks
    df = compute_greeks(df, r=args.rate, q=args.div_yield)

    # Save data
    csv_path = Path(__file__).resolve().parent / f"{args.ticker.lower()}_options_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved: {csv_path}")

    # Plot
    print("\nGenerating plots...")
    plot_vol_surface(df, args.ticker, "call")
    plot_vol_surface(df, args.ticker, "put")
    plot_smile(df, args.ticker)

    print("\nDone.")


if __name__ == "__main__":
    main()
