"""Closed-form benchmarks used to validate the Monte Carlo engine."""
from math import erfc, exp, log, pi, sqrt


class norm:
    """Standard normal cdf and pdf (no SciPy dependency)."""

    @staticmethod
    def cdf(x):
        return 0.5 * erfc(-x / sqrt(2.0))

    @staticmethod
    def pdf(x):
        return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(S0, K, T, r, sigma):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return d1, d1 - sigma * sqrt(T)


def bs_call(S0, K, T, r, sigma):
    """Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_delta(S0, K, T, r, sigma):
    return norm.cdf(_d1_d2(S0, K, T, r, sigma)[0])


def bs_gamma(S0, K, T, r, sigma):
    return norm.pdf(_d1_d2(S0, K, T, r, sigma)[0]) / (S0 * sigma * sqrt(T))


def bs_vega(S0, K, T, r, sigma):
    return S0 * norm.pdf(_d1_d2(S0, K, T, r, sigma)[0]) * sqrt(T)


def geometric_asian_call(S0, K, T, r, sigma, m):
    """Discretely monitored geometric Asian call (dates iT/m, i = 1..m). log G is Gaussian."""
    mu = log(S0) + (r - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    v = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    d1 = (mu - log(K) + v) / sqrt(v)
    d2 = d1 - sqrt(v)
    return exp(-r * T) * (exp(mu + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))


def up_and_out_call(S0, K, B, T, r, sigma):
    """Continuously monitored up-and-out call, B > K, no rebate (Hull, Options Futures and
    Other Derivatives, ch. 26): c_uo = c - c_ui."""
    if S0 >= B:
        return 0.0
    sT = sigma * sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    x1 = log(S0 / B) / sT + lam * sT
    y = log(B**2 / (S0 * K)) / sT + lam * sT
    y1 = log(B / S0) / sT + lam * sT
    c_ui = (S0 * norm.cdf(x1) - K * exp(-r * T) * norm.cdf(x1 - sT)
            - S0 * (B / S0) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
            + K * exp(-r * T) * (B / S0) ** (2 * lam - 2) * (norm.cdf(-y + sT) - norm.cdf(-y1 + sT)))
    return bs_call(S0, K, T, r, sigma) - c_ui
