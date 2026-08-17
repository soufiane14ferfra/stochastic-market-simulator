High-Performance Monte Carlo Options Pricer
Overview
This repository contains a high-performance Monte Carlo simulation engine for pricing European options. To overcome the performance bottlenecks of pure Python, the core stochastic simulation is implemented in C++, while the data manipulation and pricing logic are handled in Python using ctypes.

Mathematical Framework
1. The Asset Price Dynamics
The underlying asset is assumed to follow a Geometric Brownian Motion (GBM) under the risk-neutral measure, described by the following Stochastic Differential Equation (SDE):

dS 
t
​	
 =rS 
t
​	
 dt+σS 
t
​	
 dW 
t
​	
 
Where:
S 
t
​	
  is the asset price at time t
r is the risk-free interest rate
σ is the constant volatility
W 
t
​	
  is a standard Wiener process (Brownian motion)
2. Discretization (Euler-Maruyama)
For simulation purposes, the exact solution of the SDE is discretized over small time steps Δt:

S 
t+Δt
​	
 =S 
t
​	
 exp((r− 
2
σ 
2
 
​	
 )Δt+σ 
Δt

​	
 Z)
Where Z∼N(0,1) is a standard normal random variable.

3. Option Pricing
The price of a European Call Option with strike K and maturity T is the discounted expected payoff under the risk-neutral measure:

C 
0
​	
 =e 
−rT
 E[max(S 
T
​	
 −K,0)]
By the Law of Large Numbers, this expectation is approximated by simulating N independent price paths:

C 
0
​	
 ≈e 
−rT
  
N
1
​	
  
i=1
∑
N
​	
 max(S 
T
(i)
​	
 −K,0)
Technical Architecture
C++ Engine (engine.cpp): Handles the heavy computational load. It utilizes the std::mt19937 Mersenne Twister for robust pseudo-random number generation and directly mutates pre-allocated memory via pointers for zero-copy overhead.
Python Wrapper (random_walk.py): Acts as the user interface, allocating contiguous memory arrays via numpy and interfacing with the compiled shared library via ctypes.
