import numpy as np
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from chillprop.phases import psat_anc, rhol_anc, rhov_anc
from chillprop.core import pressure
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def debug_point():
    fluid = "Nitrogen"
    T = 121.03
    P = 1.60e5
    
    print(f"Debugging {fluid} at T={T} K, P={P} Pa")
    
    params = CH.get_params(fluid)
    
    # Check Ancillaries
    T_j = jnp.array(T)
    Psat = float(psat_anc(params, T_j))
    rho_L = float(rhol_anc(params, T_j))
    rho_V = float(rhov_anc(params, T_j))
    
    print(f"Psat (Anc): {Psat} Pa")
    print(f"rho_L (Anc): {rho_L} mol/m3")
    print(f"rho_V (Anc): {rho_V} mol/m3")
    
    rho_ideal = P / (params.R * T)
    print(f"rho_ideal: {rho_ideal} mol/m3")
    
    # Check Guess logic
    is_liquid = P > Psat
    guess = rho_L if is_liquid else rho_V
    print(f"P > Psat? {is_liquid}")
    print(f"Selected Guess: {guess} mol/m3")
    
    # Check Pressure at Guess
    P_guess = float(pressure(params, jnp.array(guess), T_j))
    print(f"P(guess): {P_guess} Pa")
    
    # Trace Solver Step by Step
    rho = guess
    print("\nSolver Trace:")
    for i in range(10):
        P_curr = float(pressure(params, jnp.array(rho), T_j))
        dP_drho = float(jax.grad(pressure, argnums=1)(params, jnp.array(rho), T_j))
        
        delta = (P_curr - P) / dP_drho
        print(f"Iter {i}: rho={rho:.4f}, P={P_curr:.2e}, dP/drho={dP_drho:.2e}, delta={delta:.4f}")
        
        rho = rho - delta
        if rho < 0:
            print(f"Iter {i+1}: rho={rho:.4f} (NEGATIVE!)")
            break

if __name__ == "__main__":
    debug_point()
