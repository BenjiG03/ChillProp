import jax
import jax.numpy as jnp
import numpy as np
from chillprop.highlevel import PropsSI

def test_density_derivative_T():
    # Test d(rho)/dT at constant P
    T0 = 300.0
    P0 = 1e6
    fluid = "Air"
    
    # We define a pure wrapper for grad
    def rho_T(T):
        # returns scalar
        return PropsSI("D", "T", T, "P", P0, fluid)
    
    # Analytical JAX gradient
    # grad returns d(rho)/dT
    grad_func = jax.grad(rho_T)
    jax_grad = float(grad_func(T0))
    
    # Finite Difference
    dT = 1e-5
    rho_plus = float(PropsSI("D", "T", T0 + dT, "P", P0, fluid))
    rho_minus = float(PropsSI("D", "T", T0 - dT, "P", P0, fluid))
    fd_grad = (rho_plus - rho_minus) / (2 * dT)
    
    # Relative error
    err = abs(jax_grad - fd_grad) / abs(fd_grad)
    print(f"JAX grad (dRho/dT): {jax_grad}")
    print(f"FD grad  (dRho/dT): {fd_grad}")
    print(f"Relative Error: {err}")
    assert err < 1e-4

def test_enthalpy_derivative_P():
    # Test dh/dP at constant T
    T0 = 400.0
    P0 = 5e6
    fluid = "Nitrogen"
    
    def h_P(P):
        return PropsSI("H", "T", T0, "P", P, fluid)
        
    grad_func = jax.grad(h_P)
    jax_grad = float(grad_func(P0))
    
    dP = 10.0 # 10 Pa
    h_plus = float(PropsSI("H", "T", T0, "P", P0 + dP, fluid))
    h_minus = float(PropsSI("H", "T", T0, "P", P0 - dP, fluid))
    fd_grad = (h_plus - h_minus) / (2 * dP)
    
    err = abs(jax_grad - fd_grad) / abs(fd_grad)
    print(f"JAX grad (dH/dP): {jax_grad}")
    print(f"FD grad  (dH/dP): {fd_grad}")
    print(f"Relative Error: {err}")
    assert err < 1e-4

if __name__ == '__main__':
    test_density_derivative_T()
    test_enthalpy_derivative_P()
