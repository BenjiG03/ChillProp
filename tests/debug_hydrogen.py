
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
from chillprop import highlevel
from chillprop.transport import viscosity

def check_hydrogen():
    fluid = "Hydrogen"
    print(f"Checking {fluid}...")
    
    params = highlevel.get_params(fluid)
    v = params.viscosity
    
    print("Viscosity Params:")
    print(f"  sigma_eta: {v.sigma_eta}")
    print(f"  epsilon_over_k: {v.epsilon_over_k}")
    
    print("Dilute Model:")
    print(v.dilute)
    
    print("Higher Order Model:")
    print(v.higher_order)
    
    if v.initial_density:
        print("Initial Density Model:")
        print(f"  b: {v.initial_density.b}")
        print(f"  t: {v.initial_density.t}")

    
    T = 300.0
    rho = 1.0 # arbitrary
    
    try:
        from chillprop.transport import viscosity_dilute, viscosity_initial_density, viscosity_higher_order
        eta0 = viscosity_dilute(params, T)
        eta_init = viscosity_initial_density(params, rho, T)
        eta_ho = viscosity_higher_order(params, rho, T)
        print(f"Dilute: {eta0}")
        print(f"Initial: {eta_init}")
        print(f"Higher Order: {eta_ho}")
        eta = eta0 + eta_init + eta_ho
        print(f"Calculated Viscosity at 300K: {eta}")
    except Exception as e:
        print(f"Error calc viscosity: {e}")

if __name__ == "__main__":
    check_hydrogen()
