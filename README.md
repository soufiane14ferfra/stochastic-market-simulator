# High-Performance Monte Carlo Options Pricer

## Overview
This repository contains a high-performance Monte Carlo simulation engine for pricing European options. To overcome the performance bottlenecks of pure Python, the core stochastic simulation is implemented in **C++**, while the data manipulation and pricing logic are handled in **Python** using `ctypes`.

## Mathematical Framework

### 1. The Asset Price Dynamics
The underlying asset is assumed to follow a Geometric Brownian Motion (GBM) under the risk-neutral measure, described by the following Stochastic Differential Equation (SDE):

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Where:
* $S_t$ is the asset price at time $t$
* $r$ is the risk-free interest rate
* $\sigma$ is the constant volatility
* $W_t$ is a standard Wiener process (Brownian motion)

### 2. Discretization (Euler-Maruyama)
For simulation purposes, the exact solution of the SDE is discretized over small time steps $\Delta t$:

$$
S_{t+\Delta t} = S_t \exp\left(\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z\right)
$$

Where $Z \sim \mathcal{N}(0,1)$ is a standard normal random variable.

### 3. Option Pricing
The price of a European Call Option with strike $K$ and maturity $T$ is the discounted expected payoff under the risk-neutral measure:

$$
C_0 = e^{-rT} \mathbb{E}[\max(S_T - K, 0)]
$$

By the Law of Large Numbers, this expectation is approximated by simulating $N$ independent price paths:

$$
C_0 \approx e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
$$

## Technical Architecture
* **C++ Engine (`engine.cpp`):** Handles the heavy computational load. It utilizes the `std::mt19937` Mersenne Twister for robust pseudo-random number generation and directly mutates pre-allocated memory via pointers for zero-copy overhead.
* **Python Wrapper (`random_walk.py`):** Acts as the user interface, allocating contiguous memory arrays via `numpy` and interfacing with the compiled shared library via `ctypes`.