"""Statistical tests of the engine against closed-form results.

Run with:  make test   (or: python3 -m pytest tests/  — or: python3 tests/test_pricer.py)
Tolerances are 4 standard errors, so a correct engine fails with probability < 1e-4 per check.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analytics as an  # noqa: E402
import mcpricer as mc  # noqa: E402

S0, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def close(est, exact, k=4.0):
    return abs(est.price - exact) <= k * est.se


def test_normal_draws_are_standard_normal():
    paths = mc.simulate_paths(1.0, 1.0, 0.5, 1.0, 1, 1_000_000, seed=11)   # r = sigma^2/2: zero drift
    z = np.log(paths[:, 1])
    assert abs(z.mean()) < 4 / 1000
    assert abs(z.var() - 1) < 4 * np.sqrt(2 / 1e6)
    assert stats.kstest(z, "norm").pvalue > 1e-3


def test_european_call_all_methods_match_black_scholes():
    ref = an.bs_call(S0, K, T, r, SIGMA)
    for method in (mc.PLAIN, mc.ANTITHETIC, mc.CONTROL, mc.ANTITHETIC_CONTROL):
        assert close(mc.european_call(S0, K, T, r, SIGMA, 1_000_000, method), ref)


def test_variance_reduction_reduces_standard_error():
    se = [mc.european_call(S0, K, T, r, SIGMA, 500_000, m).se
          for m in (mc.PLAIN, mc.ANTITHETIC, mc.CONTROL, mc.ANTITHETIC_CONTROL)]
    assert se[0] > se[1] and se[0] > se[2] > se[3]


def test_results_do_not_depend_on_thread_count():
    a = mc.european_call(S0, K, T, r, SIGMA, 300_000, threads=1)
    b = mc.european_call(S0, K, T, r, SIGMA, 300_000, threads=4)
    assert a == b


def test_greeks_match_black_scholes():
    g = mc.european_greeks(S0, K, T, r, SIGMA, 1_000_000)
    assert close(g["delta_fd"], an.bs_delta(S0, K, T, r, SIGMA))
    assert close(g["delta_pathwise"], an.bs_delta(S0, K, T, r, SIGMA))
    assert close(g["gamma_fd"], an.bs_gamma(S0, K, T, r, SIGMA))
    assert close(g["vega_fd"], an.bs_vega(S0, K, T, r, SIGMA))
    assert close(g["vega_pathwise"], an.bs_vega(S0, K, T, r, SIGMA))


def test_cpp_and_python_geometric_asian_formulas_agree():
    for m in (1, 12, 52):
        assert abs(mc._lib.geometric_asian_call(S0, K, T, r, SIGMA, m)
                   - an.geometric_asian_call(S0, K, T, r, SIGMA, m)) < 1e-12
    # with a single fixing, the geometric Asian is a European call
    assert abs(an.geometric_asian_call(S0, K, T, r, SIGMA, 1) - an.bs_call(S0, K, T, r, SIGMA)) < 1e-12


def test_asian_control_variate_is_consistent_and_much_tighter():
    plain, cv, rho = mc.asian_call(S0, K, T, r, SIGMA, 12, 200_000)
    assert abs(plain.price - cv.price) <= 4 * plain.se
    assert cv.se < plain.se / 10 and rho > 0.99


def test_barrier_brownian_bridge_removes_discretisation_bias():
    exact = an.up_and_out_call(S0, K, 130.0, T, r, SIGMA)
    naive, bridge = mc.up_and_out_call(S0, K, 130.0, T, r, SIGMA, 12, 500_000)
    assert close(bridge, exact)
    assert naive.price - exact > 10 * naive.se   # naive monitoring overprices


def test_barrier_already_knocked_out_is_worthless():
    naive, bridge = mc.up_and_out_call(140.0, K, 130.0, T, r, SIGMA, 12, 10_000)
    assert naive.price == 0.0 and bridge.price == 0.0


if __name__ == "__main__":   # lets the tests run without pytest installed
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(tests)} tests passed")
