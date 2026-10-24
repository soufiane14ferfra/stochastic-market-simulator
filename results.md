### 1. Validation against Black-Scholes (S0=K=100, T=1, r=5%, sigma=20%)
| Quantity | Value |
|---|---|
| Black-Scholes price | 10.4506 |
| Monte Carlo price (1M draws) | 10.4335 |
| 95% confidence interval | [10.4047, 10.4623] |
| Empirical coverage of the 95% CI (1,000 seeds) | 94.8% |

| Estimator | Log-log convergence slope (theory: -0.5) |
|---|---|
| Plain | -0.505 |
| Antithetic + control variate | -0.525 |


### 2. Variance reduction (European call, 1M normal draws each)
| Estimator | Price | Std error | Variance reduction | Efficiency gain (variance x run time) |
|---|---|---|---|---|
| Plain | 10.4335 | 0.0147 | x1.0 | x1.0 |
| Antithetic | 10.4462 | 0.0104 | x2.0 | x2.7 |
| Control variate (S_T) | 10.4503 | 0.0056 | x6.9 | x6.9 |
| Antithetic + control variate | 10.4521 | 0.0027 | x28.7 | x39.0 |

Antithetic pairs need only half the normal draws, so they are also cheaper to run; the control variate costs almost nothing extra.


### 3. Greeks (1M draws, common random numbers, bumps 1% of S0 and 1 vol point)
| Greek | Monte Carlo | Std error | Black-Scholes | Gap in std errors |
|---|---|---|---|---|
| Delta, finite difference | 0.63642 | 0.00057 | 0.63683 | -0.72 |
| Delta, pathwise | 0.63655 | 0.00058 | 0.63683 | -0.49 |
| Gamma, finite difference | 0.01880 | 0.00011 | 0.01876 | +0.39 |
| Vega, finite difference | 37.42654 | 0.07553 | 37.52403 | -1.29 |
| Vega, pathwise | 37.42837 | 0.07555 | 37.52403 | -1.27 |


### 4. Arithmetic Asian call (12 monthly fixings, 200k paths)
| Estimator | Price | Std error |
|---|---|---|
| Plain Monte Carlo | 6.1466 | 0.0190 |
| Geometric Asian control variate | 6.1561 | 0.00052 |


Correlation between arithmetic and geometric payoffs: 0.9996. Variance reduction: x1,314. Closed-form geometric Asian price: 5.9402.

### 5. Up-and-out call, continuous barrier B=130 (500k paths per grid)
| Steps | Naive | Brownian bridge | Closed form |
|---|---|---|---|
| 4 | 4.7571 | 3.3312 | 3.3329 |
| 12 | 4.2525 | 3.3294 | 3.3329 |
| 26 | 3.9956 | 3.3341 | 3.3329 |
| 52 | 3.8309 | 3.3493 | 3.3329 |
| 126 | 3.6499 | 3.3354 | 3.3329 |
| 252 | 3.5591 | 3.3333 | 3.3329 |
| 504 | 3.4797 | 3.3192 | 3.3329 |


### 6. Performance (European call, 10M draws, 2 CPU cores)
| Implementation | Time (best of 5) | Speed-up vs NumPy |
|---|---|---|
| NumPy, vectorised | 156 ms | x1.0 |
| C++ engine, 1 thread(s) | 122 ms | x1.3 |
| C++ engine, 2 thread(s) | 61 ms | x2.6 |


Results are identical for any number of threads: each block of 16,384 paths has its own seeded random stream.
