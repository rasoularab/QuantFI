"""End-to-end example: price a European call by Crank-Nicolson finite
differences, compare it against the closed-form Black-Scholes price, and plot
convergence as the grid is refined.

Run from the project root with:
    python examples/run_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fd_option_pricing import bs_greeks, bs_price, crank_nicolson_price, fd_greeks

S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0
OPTION_TYPE = "call"


def main() -> None:
    bs = bs_price(S0, K, r, sigma, T, OPTION_TYPE)
    fd = crank_nicolson_price(S0, K, r, sigma, T, OPTION_TYPE, M=200, N=200)

    print(f"Black-Scholes price:  {bs:.4f}")
    print(f"Crank-Nicolson price: {fd.price:.4f}  (abs. error {abs(fd.price - bs):.5f})")

    bs_g = bs_greeks(S0, K, r, sigma, T, OPTION_TYPE)
    fd_g = fd_greeks(S0, K, r, sigma, T, OPTION_TYPE, M=200, N=200)

    print("\nGreek     Black-Scholes   Finite Diff.")
    for name in ["delta", "gamma", "vega", "theta", "rho"]:
        print(f"{name:<8}  {bs_g[name]:>12.4f}   {fd_g[name]:>10.4f}")

    plot_convergence()
    plot_price_curve(fd)


def plot_convergence() -> None:
    bs = bs_price(S0, K, r, sigma, T, OPTION_TYPE)
    grid_sizes = np.unique(np.logspace(np.log10(25), np.log10(1600), 16).astype(int))

    errors = np.array(
        [abs(crank_nicolson_price(S0, K, r, sigma, T, OPTION_TYPE, M=int(m), N=int(m)).price - bs) for m in grid_sizes]
    )
    slope, intercept = np.polyfit(np.log(grid_sizes), np.log(errors), 1)
    fitted = np.exp(intercept) * grid_sizes**slope

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(grid_sizes, errors, "o", color="#1f77b4", label="Absolute error")
    ax.loglog(grid_sizes, fitted, "k--", lw=1, label=f"Fitted order $\\approx${-slope:.2f}")
    ax.set_xlabel("Grid points M (with N = M time steps)")
    ax.set_ylabel("Absolute error vs. Black-Scholes")
    ax.set_title("Crank-Nicolson convergence (Rannacher-smoothed)")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved convergence plot to {out_path}")


def plot_price_curve(fd) -> None:
    S = fd.S_grid
    mask = (S >= 0.4 * S0) & (S <= 1.8 * S0)
    payoff = np.maximum(S - K, 0.0) if OPTION_TYPE == "call" else np.maximum(K - S, 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S[mask], payoff[mask], color="black", linestyle="--", label="Terminal payoff (tau=0)")
    ax.plot(S[mask], fd.V_grid[mask], color="#1f77b4", label="Price today (tau=T)")
    ax.axvline(S0, color="#d62728", linestyle=":", lw=1, label=f"$S_0$={S0}")
    ax.set_xlabel("Underlying price $S$")
    ax.set_ylabel("Option value")
    ax.set_title(f"Crank-Nicolson solution — European {OPTION_TYPE}")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "price_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved price curve plot to {out_path}")


if __name__ == "__main__":
    main()
