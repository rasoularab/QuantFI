"""End-to-end example: price a European call, compare it against the
closed-form Black-Scholes price, and plot Monte Carlo convergence as the
number of simulated paths grows.

Run from the project root with:
    python examples/run_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mc_option_pricing import bs_greeks, bs_price, mc_greeks, mc_price

S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0
OPTION_TYPE = "call"


def main() -> None:
    bs = bs_price(S0, K, r, sigma, T, OPTION_TYPE)
    mc = mc_price(S0, K, r, sigma, T, OPTION_TYPE, n_paths=500_000, seed=42)

    print(f"Black-Scholes price:  {bs:.4f}")
    print(f"Monte Carlo price:    {mc.price:.4f}  (std err {mc.std_error:.4f})")
    print(f"95% CI:               [{mc.ci_95[0]:.4f}, {mc.ci_95[1]:.4f}]")

    bs_g = bs_greeks(S0, K, r, sigma, T, OPTION_TYPE)
    mc_g = mc_greeks(S0, K, r, sigma, T, OPTION_TYPE, n_paths=500_000, seed=42)

    print("\nGreek     Black-Scholes   Monte Carlo")
    for name in ["delta", "gamma", "vega", "theta", "rho"]:
        print(f"{name:<8}  {bs_g[name]:>12.4f}   {mc_g[name]:>10.4f}")

    plot_convergence()


def plot_convergence() -> None:
    bs = bs_price(S0, K, r, sigma, T, OPTION_TYPE)
    path_counts = np.unique(np.logspace(2, 6, 25).astype(int))

    prices, half_widths = [], []
    for n in path_counts:
        result = mc_price(S0, K, r, sigma, T, OPTION_TYPE, n_paths=int(n), seed=42)
        prices.append(result.price)
        half_widths.append(1.96 * result.std_error)

    prices = np.array(prices)
    half_widths = np.array(half_widths)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(path_counts, prices, label="Monte Carlo price", color="#1f77b4")
    ax.fill_between(
        path_counts, prices - half_widths, prices + half_widths,
        alpha=0.25, color="#1f77b4", label="95% CI",
    )
    ax.axhline(bs, color="black", linestyle="--", label="Black-Scholes price")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Option price")
    ax.set_title(f"Monte Carlo convergence — European {OPTION_TYPE}")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved convergence plot to {out_path}")


if __name__ == "__main__":
    main()
