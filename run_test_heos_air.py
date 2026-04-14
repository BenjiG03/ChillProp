import sys
import jax
import jax.numpy as jnp
from chillprop.parameters import FluidParameters
from chillprop.heos import evaluate_alpha0

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

def main():
    import json
    with open('chillprop/data/Air.json', 'r') as f:
        data = json.load(f)
    params = FluidParameters.from_json(data)
    
    T = 300.0
    rho_mass = 1.2 # kg/m^3
    rho = rho_mass / params.M
    
    alpha0_cp = CP.PropsSI("alpha0", "T", T, "D", rho_mass, "Air")
    
    # Evaluate chillprop alpha0
    tau = params.Tr / T
    delta = rho / params.rhor
    
    # To isolate the generalized planck einstein term
    # we can see what chillprop produces
    try:
        a0_cp = evaluate_alpha0(params, rho, T)
        print(f"CoolProp alpha0: {alpha0_cp}")
        print(f"ChillProp alpha0: {a0_cp}")
    except Exception as e:
        print(f"Error evaluating: {e}")

if __name__ == '__main__':
    main()
