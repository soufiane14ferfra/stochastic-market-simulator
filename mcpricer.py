"""Python interface to the C++ Monte Carlo engine (ctypes).

Build the library first:  make build   (or: g++ -O3 -std=c++17 -pthread -shared -fPIC engine.cpp -o libengine.so)
"""
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_lib = ctypes.CDLL(str(Path(__file__).resolve().with_name("libengine.so")))

D, L, I, U64 = ctypes.c_double, ctypes.c_long, ctypes.c_int, ctypes.c_uint64
P = ctypes.POINTER(ctypes.c_double)

_lib.simulate_gbm.argtypes = [D, D, D, D, I, I, U64, P]
_lib.european_call.argtypes = [D, D, D, D, D, L, U64, I, I, P]
_lib.european_greeks.argtypes = [D, D, D, D, D, L, D, D, U64, I, P]
_lib.asian_call.argtypes = [D, D, D, D, D, I, L, U64, I, P]
_lib.up_and_out_call.argtypes = [D, D, D, D, D, D, I, L, U64, I, P]
_lib.geometric_asian_call.argtypes = [D, D, D, D, D, I]
_lib.geometric_asian_call.restype = D
for f in ("simulate_gbm", "european_call", "european_greeks", "asian_call", "up_and_out_call"):
    getattr(_lib, f).restype = None

DEFAULT_THREADS = os.cpu_count() or 1

PLAIN, ANTITHETIC, CONTROL, ANTITHETIC_CONTROL = 0, 1, 2, 3


@dataclass
class Estimate:
    price: float
    se: float  # standard error of the estimator

    @property
    def ci95(self):
        return self.price - 1.96 * self.se, self.price + 1.96 * self.se


def _out(n):
    return (ctypes.c_double * n)()


def simulate_paths(S0, T, r, sigma, n_steps, n_sims, seed=0):
    """GBM paths, shape (n_sims, n_steps + 1). The C++ engine fills NumPy's buffer directly."""
    paths = np.empty((n_sims, n_steps + 1))
    _lib.simulate_gbm(S0, T, r, sigma, n_steps, n_sims, seed, paths.ctypes.data_as(P))
    return paths


def european_call(S0, K, T, r, sigma, n_draws, method=PLAIN, seed=42, threads=DEFAULT_THREADS):
    """n_draws = number of normal draws. Antithetic uses them as n_draws/2 pairs."""
    o = _out(3)
    _lib.european_call(S0, K, T, r, sigma, n_draws, seed, method, threads, o)
    return Estimate(o[0], o[1])


def european_greeks(S0, K, T, r, sigma, n_draws, hS=0.01, hV=0.01, seed=42, threads=DEFAULT_THREADS):
    o = _out(10)
    _lib.european_greeks(S0, K, T, r, sigma, n_draws, hS, hV, seed, threads, o)
    names = ["delta_fd", "gamma_fd", "vega_fd", "delta_pathwise", "vega_pathwise"]
    return {n: Estimate(o[2 * k], o[2 * k + 1]) for k, n in enumerate(names)}


def asian_call(S0, K, T, r, sigma, m, n_paths, seed=42, threads=DEFAULT_THREADS):
    """Arithmetic Asian call: (plain, control variate, correlation with the geometric Asian)."""
    o = _out(5)
    _lib.asian_call(S0, K, T, r, sigma, m, n_paths, seed, threads, o)
    return Estimate(o[0], o[1]), Estimate(o[2], o[3]), o[4]


def up_and_out_call(S0, K, B, T, r, sigma, m, n_paths, seed=42, threads=DEFAULT_THREADS):
    """Continuously monitored up-and-out call on an m-step grid: (naive, Brownian bridge)."""
    o = _out(4)
    _lib.up_and_out_call(S0, K, B, T, r, sigma, m, n_paths, seed, threads, o)
    return Estimate(o[0], o[1]), Estimate(o[2], o[3])
