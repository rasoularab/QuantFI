import numpy as np
import pytest

from fd_option_pricing import bs_greeks, bs_price, crank_nicolson_price, fd_greeks

PARAMS = dict(S0=100.0, K=100.0, r=0.02, sigma=0.2, T=1.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fd_price_matches_black_scholes(option_type):
    bs = bs_price(option_type=option_type, **PARAMS)
    fd = crank_nicolson_price(option_type=option_type, M=200, N=200, **PARAMS)
    assert fd.price == pytest.approx(bs, abs=1e-2)


def test_convergence_order_is_approximately_second_order():
    bs = bs_price(option_type="call", **PARAMS)
    grid_sizes = [50, 100, 200, 400, 800, 1600]
    errors = [abs(crank_nicolson_price(option_type="call", M=m, N=m, **PARAMS).price - bs) for m in grid_sizes]

    # Individual points are noisy (the payoff's kink at the strike pollutes
    # any single grid's error), so fit the order across many grid sizes
    # rather than comparing two points directly.
    slope, _ = np.polyfit(np.log(grid_sizes), np.log(errors), 1)
    order = -slope
    assert order > 1.3


def test_put_call_parity_holds():
    call = crank_nicolson_price(option_type="call", M=200, N=200, **PARAMS)
    put = crank_nicolson_price(option_type="put", M=200, N=200, **PARAMS)

    S0, K, r, T = PARAMS["S0"], PARAMS["K"], PARAMS["r"], PARAMS["T"]
    parity_lhs = call.price - put.price
    parity_rhs = S0 - K * np.exp(-r * T)

    assert parity_lhs == pytest.approx(parity_rhs, abs=1e-2)


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_fd_greeks_match_black_scholes(greek):
    bs_g = bs_greeks(option_type="call", **PARAMS)
    fd_g = fd_greeks(option_type="call", M=200, N=200, **PARAMS)
    assert fd_g[greek] == pytest.approx(bs_g[greek], rel=0.02, abs=0.05)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        crank_nicolson_price(option_type="straddle", M=50, N=50, **PARAMS)


def test_grid_S0_is_exact_node():
    fd = crank_nicolson_price(option_type="call", M=200, N=200, **PARAMS)
    assert np.min(np.abs(fd.S_grid - PARAMS["S0"])) < 1e-9


def test_terminal_boundary_conditions_are_asymptotically_correct():
    fd = crank_nicolson_price(option_type="call", M=200, N=200, **PARAMS)
    assert fd.V_grid[0] == pytest.approx(0.0, abs=1e-6)
    assert fd.V_grid[-1] > fd.S_grid[-1] - PARAMS["K"]
