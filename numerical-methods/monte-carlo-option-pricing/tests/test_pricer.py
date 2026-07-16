import numpy as np
import pytest

from mc_option_pricing import bs_greeks, bs_price, mc_greeks, mc_price, simulate_terminal_prices

PARAMS = dict(S0=100.0, K=100.0, r=0.02, sigma=0.2, T=1.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_converges_to_black_scholes(option_type):
    bs = bs_price(option_type=option_type, **PARAMS)
    mc = mc_price(option_type=option_type, n_paths=200_000, seed=1, **PARAMS)
    assert mc.price == pytest.approx(bs, abs=3 * mc.std_error)


def test_confidence_interval_contains_price():
    mc = mc_price(option_type="call", n_paths=50_000, seed=2, **PARAMS)
    lo, hi = mc.ci_95
    assert lo < mc.price < hi


def test_antithetic_variates_reduce_variance():
    kwargs = dict(option_type="call", n_paths=50_000, seed=3, **PARAMS)
    plain = mc_price(antithetic=False, **kwargs)
    anti = mc_price(antithetic=True, **kwargs)
    assert anti.std_error < plain.std_error


def test_put_call_parity_holds_in_expectation():
    call = mc_price(option_type="call", n_paths=200_000, seed=4, **PARAMS)
    put = mc_price(option_type="put", n_paths=200_000, seed=4, **PARAMS)

    S0, K, r, T = PARAMS["S0"], PARAMS["K"], PARAMS["r"], PARAMS["T"]
    parity_lhs = call.price - put.price
    parity_rhs = S0 - K * np.exp(-r * T)
    combined_se = np.sqrt(call.std_error**2 + put.std_error**2)

    assert parity_lhs == pytest.approx(parity_rhs, abs=3 * combined_se)


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_mc_greeks_match_black_scholes(greek):
    bs_g = bs_greeks(option_type="call", **PARAMS)
    mc_g = mc_greeks(option_type="call", n_paths=300_000, seed=5, **PARAMS)
    assert mc_g[greek] == pytest.approx(bs_g[greek], rel=0.05, abs=0.1)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_price(option_type="straddle", n_paths=1_000, **PARAMS)


def test_simulate_terminal_prices_shape_and_positivity():
    rng = np.random.default_rng(7)
    S_T = simulate_terminal_prices(S0=100, r=0.02, sigma=0.2, T=1.0, n_paths=1_001, rng=rng)
    assert S_T.shape == (1_001,)
    assert np.all(S_T > 0)
