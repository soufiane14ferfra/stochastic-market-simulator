"""Etape 1 : pricer Monte Carlo (moteur C++ via ctypes) valide contre Black-Scholes."""
import ctypes
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Chargement du moteur C++ -------------------------------------------------
# Compilation : g++ -O3 -shared -fPIC engine.cpp -o libengine.so
lib = ctypes.CDLL(str(Path(__file__).with_name("libengine.so")))
lib.price_european_call.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_long, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double),
]
lib.price_european_call.restype = None


def mc_call(S0, K, T, r, sigma, n_sims, seed=42):
    """Retourne (prix, erreur standard) calcules par le moteur C++."""
    out = (ctypes.c_double * 2)()
    lib.price_european_call(S0, K, T, r, sigma, n_sims, seed, out)
    return out[0], out[1]


def bs_call(S0, K, T, r, sigma):
    """Formule fermee de Black-Scholes pour un call europeen."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S0 * N(d1) - K * math.exp(-r * T) * N(d2)


def numpy_call(S0, K, T, r, sigma, n_sims, seed=42):
    """Version NumPy vectorisee, pour un benchmark honnete."""
    Z = np.random.default_rng(seed).standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    payoff = math.exp(-r * T) * np.maximum(ST - K, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / math.sqrt(n_sims)


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    ref = bs_call(S0, K, T, r, sigma)

    # 1) Validation contre Black-Scholes avec intervalle de confiance a 95 %
    price, se = mc_call(S0, K, T, r, sigma, 1_000_000)
    lo, hi = price - 1.96 * se, price + 1.96 * se
    print(f"Black-Scholes : {ref:.4f}")
    print(f"Monte Carlo   : {price:.4f}  (IC 95 % : [{lo:.4f}, {hi:.4f}])")
    print(f"BS dans l'IC  : {lo <= ref <= hi}")

    # 2) Couverture de l'IC : sur 200 graines, ~95 % des IC doivent contenir le vrai prix
    hits = 0
    for s in range(200):
        p, e = mc_call(S0, K, T, r, sigma, 10_000, seed=s)
        hits += abs(p - ref) <= 1.96 * e
    print(f"Couverture empirique de l'IC : {hits / 200:.1%} (attendu ~95 %)")

    # 3) Convergence : l'erreur RMS doit decroitre en 1/sqrt(N) (pente -1/2 en log-log)
    Ns = np.logspace(2, 6, 9).astype(int)
    rmse = []
    for N in Ns:
        errs = [mc_call(S0, K, T, r, sigma, int(N), seed=s)[0] - ref for s in range(50)]
        rmse.append(math.sqrt(np.mean(np.square(errs))))
    slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]
    print(f"Pente de convergence log-log : {slope:.3f} (theorie : -0.5)")

    plt.figure(figsize=(7, 4.5))
    plt.loglog(Ns, rmse, "o-", label="Erreur RMS observee (50 graines)")
    plt.loglog(Ns, rmse[0] * np.sqrt(Ns[0] / Ns), "--", label=r"Theorie $\propto 1/\sqrt{N}$")
    plt.xlabel("Nombre de simulations N")
    plt.ylabel("Erreur RMS vs Black-Scholes")
    plt.title(f"Convergence du pricer Monte Carlo (pente = {slope:.2f})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)

    # 4) Benchmark honnete C++ vs NumPy (meilleur de 5 runs)
    N = 1_000_000
    def best(f):
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); f(); times.append(time.perf_counter() - t0)
        return min(times)
    t_cpp = best(lambda: mc_call(S0, K, T, r, sigma, N))
    t_np = best(lambda: numpy_call(S0, K, T, r, sigma, N))
    print(f"1M trajectoires : C++ {t_cpp*1e3:.1f} ms | NumPy {t_np*1e3:.1f} ms | ratio {t_np/t_cpp:.2f}x")
