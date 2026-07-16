"""Finite-difference pricing of European options via the Black-Scholes PDE.

Under Black-Scholes-Merton, V(S, t) satisfies

    dV/dt + 0.5 sigma^2 S^2 d2V/dS2 + r S dV/dS - r V = 0,   V(S, T) = payoff(S)

Substituting the backward time variable tau = T - t turns this into an
initial value problem in tau, starting from the known payoff at tau=0 and
marching forward to tau=T (i.e. to today's price). Each step uses a
theta-scheme: theta=0.5 is Crank-Nicolson (unconditionally stable, second
order in both S and tau); theta=1 is fully implicit (first order, but more
strongly damps high-frequency error).

The payoff's kink at the strike is a discontinuity in dV/dS that pollutes
Crank-Nicolson's clean second-order convergence with small spurious
oscillations for the first few steps. The standard fix — Rannacher
smoothing — starts the march with a couple of fully implicit steps before
switching to Crank-Nicolson, which damps that high-frequency error without
giving up second-order convergence overall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class FDResult:
    price: float
    delta: float
    gamma: float
    theta: float
    S_grid: np.ndarray
    V_grid: np.ndarray


def _boundary_values(S_max: float, K: float, r: float, tau: float, option_type: str) -> tuple[float, float]:
    """Option value at S=0 and S=S_max, at backward time tau (i.e. real time T - tau)."""
    if option_type == "call":
        return 0.0, S_max - K * np.exp(-r * tau)
    if option_type == "put":
        return K * np.exp(-r * tau), 0.0
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _build_theta_scheme(
    theta_used: float, dtau: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, M: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Banded implicit-side matrix and explicit-side coefficient arrays for one
    theta-scheme time step (theta_used=0.5 is Crank-Nicolson, 1.0 is fully implicit).
    """
    A = theta_used * dtau * a
    B = theta_used * dtau * b
    G = theta_used * dtau * c
    Ae = (1 - theta_used) * dtau * a
    Be = (1 - theta_used) * dtau * b
    Ge = (1 - theta_used) * dtau * c

    interior = slice(1, M)
    n_interior = M - 1
    ab = np.zeros((3, n_interior))
    ab[0, 1:] = -G[1 : M - 1]  # super-diagonal
    ab[1, :] = 1.0 - B[interior]  # main diagonal
    ab[2, :-1] = -A[2:M]  # sub-diagonal

    return ab, A, Ae, G, Be, Ge


def crank_nicolson_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    S_max: float | None = None,
    M: int = 200,
    N: int = 200,
    rannacher_steps: int = 2,
) -> FDResult:
    """Price a European option by solving the Black-Scholes PDE with a
    Crank-Nicolson finite-difference scheme on a uniform grid in the
    underlying price (with Rannacher smoothing at the start of the march).

    The grid spacing is chosen so that S0 falls exactly on a node. That means
    the price needs no interpolation, and delta/gamma can be read directly
    off the grid via central differences on the neighboring nodes.
    """
    S_max = S_max or 4.0 * max(S0, K)
    n0 = max(1, round(S0 / (S_max / M)))
    dS = S0 / n0
    M = round(S_max / dS)
    dtau = T / N
    i = np.arange(0, M + 1)
    S = i * dS

    if option_type == "call":
        V = np.maximum(S - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # PDE coefficients (independent of theta and of tau, only depend on S via i).
    a = 0.5 * sigma**2 * i**2 - 0.5 * r * i
    b = -sigma**2 * i**2 - r
    c = 0.5 * sigma**2 * i**2 + 0.5 * r * i

    interior = slice(1, M)
    scheme_implicit = _build_theta_scheme(1.0, dtau, a, b, c, M)
    scheme_cn = _build_theta_scheme(0.5, dtau, a, b, c, M)

    V_prev_step = V.copy()
    rannacher_steps = min(rannacher_steps, N)

    for n in range(N):
        implicit_step = n < rannacher_steps
        ab, A, Ae, G, Be, Ge = scheme_implicit if implicit_step else scheme_cn

        tau_next = (n + 1) * dtau
        V0_next, VM_next = _boundary_values(S_max, K, r, tau_next, option_type)

        d = Ae[interior] * V[: M - 1] + (1 + Be[interior]) * V[interior] + Ge[interior] * V[2 : M + 1]
        d[0] += A[1] * V0_next
        d[-1] += G[M - 1] * VM_next

        if n == N - 1:
            V_prev_step = V.copy()

        V_interior = solve_banded((1, 1), ab, d)
        V = np.empty(M + 1)
        V[0], V[-1] = V0_next, VM_next
        V[interior] = V_interior

    price = float(V[n0])
    delta = float((V[n0 + 1] - V[n0 - 1]) / (2 * dS))
    gamma = float((V[n0 + 1] - 2 * V[n0] + V[n0 - 1]) / dS**2)
    theta = float(-(V[n0] - V_prev_step[n0]) / dtau)

    return FDResult(price=price, delta=delta, gamma=gamma, theta=theta, S_grid=S, V_grid=V)


def fd_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    S_max: float | None = None,
    M: int = 200,
    N: int = 200,
    bump: dict[str, float] | None = None,
) -> dict[str, float]:
    """All five Greeks for a Crank-Nicolson price.

    Delta, gamma and theta come directly off the single base grid. Vega and
    rho are obtained by central finite differences on the base solve (i.e.
    two extra PDE solves each), which is exact up to discretization error
    since the solver is fully deterministic — unlike Monte Carlo, there is no
    simulation noise to average out.
    """
    bump = bump or {"sigma": 1e-3, "r": 1e-4}
    base = crank_nicolson_price(S0, K, r, sigma, T, option_type, S_max, M, N)

    h_sigma = bump["sigma"]
    up = crank_nicolson_price(S0, K, r, sigma + h_sigma, T, option_type, S_max, M, N)
    down = crank_nicolson_price(S0, K, r, sigma - h_sigma, T, option_type, S_max, M, N)
    vega = (up.price - down.price) / (2 * h_sigma)

    h_r = bump["r"]
    up_r = crank_nicolson_price(S0, K, r + h_r, sigma, T, option_type, S_max, M, N)
    down_r = crank_nicolson_price(S0, K, r - h_r, sigma, T, option_type, S_max, M, N)
    rho = (up_r.price - down_r.price) / (2 * h_r)

    return {"delta": base.delta, "gamma": base.gamma, "vega": vega, "theta": base.theta, "rho": rho}
