import jax.numpy as jnp
import jax

def evaluate_ancillary_test(n, t, T_r, T, reducing_value, using_tau_r):
    theta = 1.0 - T / T_r
    sigma = jnp.sum(n * (theta ** t))
    exponent = jnp.where(using_tau_r, sigma * (T_r / T), sigma)
    val = reducing_value * jnp.exp(exponent)
    return val, theta, sigma, exponent

n = jnp.array([0.2260724, -7.080499, 5.700283, -12.44017, 17.81926, -10.81364])
t = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
T_r = 132.6312
reducing_value = 3785020.0
T = 100.0
using_tau_r = True

res = evaluate_ancillary_test(n, t, T_r, T, reducing_value, using_tau_r)
print(f"Result: {res[0]}")
print(f"Theta: {res[1]}")
print(f"Sigma: {res[2]}")
print(f"Exponent: {res[3]}")
