"""Reproduces every number and figure in the README.

Usage:  make experiments   (or: python3 run_experiments.py)
Writes figures/*.png and results.md.
"""
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analytics as an
import mcpricer as mc

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.figsize": (7, 4.5), "axes.grid": True, "grid.alpha": 0.3})
report = []


def section(title):
    print(f"\n== {title}")
    report.append(f"\n### {title}\n")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    print("\n".join(lines))
    report.append("\n".join(lines) + "\n\n")


def best_time(f, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


# --- 0. Sample paths ----------------------------------------------------------
paths = mc.simulate_paths(S0, T, r, sigma, 252, 20, seed=7)
plt.plot(np.arange(253), paths.T, lw=0.8)
plt.title("20 GBM paths simulated by the C++ engine (exact scheme, 252 steps)")
plt.xlabel("Trading day")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(FIG / "paths.png", dpi=150)
plt.close()

# --- 1. Validation against Black-Scholes -------------------------------------
section("1. Validation against Black-Scholes (S0=K=100, T=1, r=5%, sigma=20%)")
ref = an.bs_call(S0, K, T, r, sigma)
est = mc.european_call(S0, K, T, r, sigma, 1_000_000)
lo, hi = est.ci95
small = [mc.european_call(S0, K, T, r, sigma, 10_000, seed=s) for s in range(1000)]
cover = np.mean([abs(e.price - ref) <= 1.96 * e.se for e in small])
table(["Quantity", "Value"], [
    ["Black-Scholes price", f"{ref:.4f}"],
    ["Monte Carlo price (1M draws)", f"{est.price:.4f}"],
    ["95% confidence interval", f"[{lo:.4f}, {hi:.4f}]"],
    ["Empirical coverage of the 95% CI (1,000 seeds)", f"{cover:.1%}"],
])

Ns = np.logspace(2, 6, 9).astype(int)
methods = {"Plain": mc.PLAIN, "Antithetic + control variate": mc.ANTITHETIC_CONTROL}
slopes = {}
for label, m in methods.items():
    rmse = [math.sqrt(np.mean([(mc.european_call(S0, K, T, r, sigma, int(N), m, seed=s).price - ref) ** 2
                               for s in range(100)])) for N in Ns]
    slopes[label] = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]
    plt.loglog(Ns, rmse, "o-", label=f"{label} (slope {slopes[label]:.2f})")
plt.loglog(Ns, 2 * Ns[0] ** 0.5 / np.sqrt(Ns), "k--", lw=1, label=r"$\propto 1/\sqrt{N}$")
plt.xlabel("Number of normal draws N")
plt.ylabel("RMS error vs Black-Scholes (100 seeds)")
plt.title("Convergence of the Monte Carlo estimator")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "convergence.png", dpi=150)
plt.close()
table(["Estimator", "Log-log convergence slope (theory: -0.5)"],
      [[k, f"{v:.3f}"] for k, v in slopes.items()])

# --- 2. Variance reduction ------------------------------------------------------
section("2. Variance reduction (European call, 1M normal draws each)")
N = 1_000_000
base = mc.european_call(S0, K, T, r, sigma, N, mc.PLAIN)
variants = [("Plain", mc.PLAIN), ("Antithetic", mc.ANTITHETIC),
            ("Control variate (S_T)", mc.CONTROL), ("Antithetic + control variate", mc.ANTITHETIC_CONTROL)]
# single-threaded timings, best of 15, so that run-time ratios are stable
times = {m: best_time(lambda: mc.european_call(S0, K, T, r, sigma, N, m, threads=1), repeat=15)
         for _, m in variants}
rows = []
for label, m in variants:
    e = mc.european_call(S0, K, T, r, sigma, N, m)
    vr = (base.se / e.se) ** 2
    rows.append([label, f"{e.price:.4f}", f"{e.se:.4f}", f"x{vr:.1f}",
                 f"x{vr * times[mc.PLAIN] / times[m]:.1f}"])
table(["Estimator", "Price", "Std error", "Variance reduction",
       "Efficiency gain (variance x run time)"], rows)
report.append("Antithetic pairs need only half the normal draws, so they are also cheaper to run; "
              "the control variate costs almost nothing extra.\n\n")

# --- 3. Greeks --------------------------------------------------------------
section("3. Greeks (1M draws, common random numbers, bumps 1% of S0 and 1 vol point)")
g = mc.european_greeks(S0, K, T, r, sigma, 1_000_000)
exact = {"delta": an.bs_delta(S0, K, T, r, sigma), "gamma": an.bs_gamma(S0, K, T, r, sigma),
         "vega": an.bs_vega(S0, K, T, r, sigma)}
rows = []
for key, label in [("delta_fd", "Delta, finite difference"), ("delta_pathwise", "Delta, pathwise"),
                   ("gamma_fd", "Gamma, finite difference"), ("vega_fd", "Vega, finite difference"),
                   ("vega_pathwise", "Vega, pathwise")]:
    e, x = g[key], exact[key.split("_")[0]]
    rows.append([label, f"{e.price:.5f}", f"{e.se:.5f}", f"{x:.5f}", f"{(e.price - x) / e.se:+.2f}"])
table(["Greek", "Monte Carlo", "Std error", "Black-Scholes", "Gap in std errors"], rows)

spots = np.linspace(70, 130, 25)
d_mc = [mc.european_greeks(s, K, T, r, sigma, 200_000)["delta_pathwise"].price for s in spots]
fine = np.linspace(70, 130, 300)
plt.plot(fine, [an.bs_delta(s, K, T, r, sigma) for s in fine], "k-", lw=1, label="Black-Scholes")
plt.plot(spots, d_mc, "o", ms=4, label="Monte Carlo, pathwise (200k draws)")
plt.xlabel("Spot S0")
plt.ylabel("Delta")
plt.title("Call delta across spots")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "delta.png", dpi=150)
plt.close()

# --- 4. Asian option --------------------------------------------------------------
section("4. Arithmetic Asian call (12 monthly fixings, 200k paths)")
plain, cv, rho = mc.asian_call(S0, K, T, r, sigma, 12, 200_000)
table(["Estimator", "Price", "Std error"], [
    ["Plain Monte Carlo", f"{plain.price:.4f}", f"{plain.se:.4f}"],
    ["Geometric Asian control variate", f"{cv.price:.4f}", f"{cv.se:.5f}"],
])
report.append(f"\nCorrelation between arithmetic and geometric payoffs: {rho:.4f}. "
              f"Variance reduction: x{(plain.se / cv.se) ** 2:,.0f}. Closed-form geometric Asian price: "
              f"{an.geometric_asian_call(S0, K, T, r, sigma, 12):.4f}.\n")

# --- 5. Barrier option -------------------------------------------------------------
section("5. Up-and-out call, continuous barrier B=130 (500k paths per grid)")
B = 130.0
exact_b = an.up_and_out_call(S0, K, B, T, r, sigma)
grid = [4, 12, 26, 52, 126, 252, 504]
naive, bridge = zip(*[mc.up_and_out_call(S0, K, B, T, r, sigma, m, 500_000, seed=m) for m in grid])
plt.axhline(exact_b, color="k", lw=1, label=f"Closed form, continuous barrier ({exact_b:.3f})")
plt.errorbar(grid, [e.price for e in naive], [1.96 * e.se for e in naive], fmt="o-", capsize=3,
             label="Naive: check barrier on grid dates only")
plt.errorbar(grid, [e.price for e in bridge], [1.96 * e.se for e in bridge], fmt="s-", capsize=3,
             label="Brownian bridge correction")
plt.xscale("log")
plt.xlabel("Number of monitoring steps m")
plt.ylabel("Price")
plt.title("Discretisation bias of a barrier option")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "barrier.png", dpi=150)
plt.close()
table(["Steps", "Naive", "Brownian bridge", "Closed form"],
      [[m, f"{a.price:.4f}", f"{b.price:.4f}", f"{exact_b:.4f}"] for m, a, b in zip(grid, naive, bridge)])

# --- 6. Performance -----------------------------------------------------------------
section(f"6. Performance (European call, 10M draws, {os.cpu_count()} CPU cores)")
N = 10_000_000


def numpy_call(n, chunk=1_000_000):
    rng, total = np.random.default_rng(0), 0.0
    for _ in range(n // chunk):
        z = rng.standard_normal(chunk)
        total += np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) - K, 0).sum()
    return math.exp(-r * T) * total / n


t_np = best_time(lambda: numpy_call(N))
rows = [["NumPy, vectorised", f"{t_np * 1e3:.0f} ms", "x1.0"]]
for th in sorted({1, 2, os.cpu_count() or 1}):
    t = best_time(lambda: mc.european_call(S0, K, T, r, sigma, N, threads=th))
    rows.append([f"C++ engine, {th} thread(s)", f"{t * 1e3:.0f} ms", f"x{t_np / t:.1f}"])
table(["Implementation", "Time (best of 5)", "Speed-up vs NumPy"], rows)
report.append("\nResults are identical for any number of threads: each block of 16,384 paths "
              "has its own seeded random stream.\n")

Path("results.md").write_text("".join(report).lstrip())
print("\nWrote results.md and figures/")
