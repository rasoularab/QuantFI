"""Monte Carlo pricing of European options under geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_95: tuple[float, float]
    n_paths: int


def simulate_terminal_prices(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    antithetic: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw terminal stock prices under the risk-neutral GBM measure.

    S_T = S0 * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z), Z ~ N(0, 1)

    The exact one-step distribution is used rather than stepping through
    intermediate time points, since a European payoff only depends on S_T.
    """
    rng = rng if rng is not None else np.random.default_rng()
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    return S0 * np.exp(drift + diffusion * z)


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def mc_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Price a European option by Monte Carlo simulation.

    Returns the discounted-payoff sample mean along with its standard error
    and a 95% confidence interval, so the estimate is reported with its
    uncertainty rather than as a bare number.
    """
    rng = np.random.default_rng(seed)
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_paths, antithetic, rng)
    discounted_payoffs = np.exp(-r * T) * _payoff(S_T, K, option_type)

    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(n_paths))
    ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

    return MCResult(price=price, std_error=std_error, ci_95=ci_95, n_paths=n_paths)


def mc_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 0,
    bump: dict[str, float] | None = None,
) -> dict[str, float]:
    """Estimate Greeks by central finite differences.

    Each bumped scenario reuses the same seed, so the bumped and base runs
    share the same underlying draws (common random numbers). This makes the
    *difference* between runs driven by the bump itself rather than by fresh
    simulation noise, which is what makes finite-difference Greeks usable at
    a moderate path count.
    """
    bump = bump or {"S0": 0.01 * S0, "sigma": 1e-4, "T": 1 / 365, "r": 1e-4}
    h_S, h_sigma, h_T, h_r = bump["S0"], bump["sigma"], bump["T"], bump["r"]

    def price_at(**overrides: float) -> float:
        params = {"S0": S0, "r": r, "sigma": sigma, "T": T}
        params.update(overrides)
        rng = np.random.default_rng(seed)
        S_T = simulate_terminal_prices(
            params["S0"], params["r"], params["sigma"], params["T"], n_paths, True, rng
        )
        discounted = np.exp(-params["r"] * params["T"]) * _payoff(S_T, K, option_type)
        return float(discounted.mean())

    base = price_at()
    delta = (price_at(S0=S0 + h_S) - price_at(S0=S0 - h_S)) / (2 * h_S)
    gamma = (price_at(S0=S0 + h_S) - 2 * base + price_at(S0=S0 - h_S)) / (h_S**2)
    vega = (price_at(sigma=sigma + h_sigma) - price_at(sigma=sigma - h_sigma)) / (2 * h_sigma)
    theta = -(price_at(T=T + h_T) - price_at(T=T - h_T)) / (2 * h_T)
    rho = (price_at(r=r + h_r) - price_at(r=r - h_r)) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
