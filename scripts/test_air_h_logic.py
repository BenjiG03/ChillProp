import jax.numpy as jnp
import numpy as np

# R for Air
R = 8.31451 
Tr = 132.6312
M = 0.02896546
T = 300
tau = Tr / T

# From Air.json IdealGasHelmholtzPower
n_paper = 17.275266575
t_paper = 1

# From Air.json IdealGasHelmholtzEnthalpyEntropyOffset
a2_offset = 3.31110091060965

# LogTau term
a1_logtau = 2.5 

# Enthalpy calculation (Ideal Gas part only)
# h^0 = R * T * (1 + tau * alpha0_tau)
# alpha0_tau = sum(n_i * t_i * tau^(t_i-1))

# Case 1: Sum EVERYTHING (ChillProp current)
alpha0_tau_1 = (n_paper * t_paper * (tau**(t_paper-1))) + (a2_offset) + (a1_logtau / tau)
h0_1 = R * T * (1 + tau * alpha0_tau_1) / M
print(f"Case 1 (Sum all): {h0_1}")

# Case 2: Use only Offset a2 (Ignore Power t=1)
alpha0_tau_2 = (a2_offset) + (a1_logtau / tau)
h0_2 = R * T * (1 + tau * alpha0_tau_2) / M
print(f"Case 2 (Ignore Power t=1): {h0_2}")

# Case 3: CoolProp actual
# 426297
print(f"CoolProp Actual: 426297")
