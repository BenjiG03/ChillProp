import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
# from chillprop import convert, props  <-- Unused and causing import error
from chillprop.highlevel import PropsSI
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def verify_fluid(fluid_name):
    print(f"Verifying {fluid_name}...")
    
    # Range of conditions
    # Try to cover liquid, gas, supercritical
    # Get critical point from CoolProp
    Tc = CP.PropsSI("Tcrit", fluid_name)
    Pc = CP.PropsSI("Pcrit", fluid_name)
    
    T_min = CP.PropsSI("Tmin", fluid_name)
    T_max = CP.PropsSI("Tmax", fluid_name)
    
    # Avoid very low T where CoolProp might be unstable or outside EOS range for transport
    T_vals = np.linspace(max(T_min, 100), min(T_max, 1000), 5)
    P_vals = np.logspace(5, 8, 5) # 1 bar to 1000 bar
    
    T_grid, P_grid = np.meshgrid(T_vals, P_vals)
    T_flat = T_grid.flatten()
    P_flat = P_grid.flatten()
    
    errors = {
        'D': [],
        'A': [],
        'V': [],
        'L': []
    }
    
    for T, P in zip(T_flat, P_flat):
        try:
            # CoolProp
            rho_cp = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            a_cp = CP.PropsSI('A', 'T', T, 'P', P, fluid_name)
            visc_cp = CP.PropsSI('V', 'T', T, 'P', P, fluid_name)
            cond_cp = CP.PropsSI('L', 'T', T, 'P', P, fluid_name)
            
            # ChillProp
            # We need to compute Density first because ChillProp low-level takes Density
            # But highlevel PropsSI takes T, P.
            # Let's use ChillProp's PropsSI to test the solver too.
            rho_chill = PropsSI('D', 'T', T, 'P', P, fluid_name)
            a_chill = PropsSI('A', 'T', T, 'P', P, fluid_name)
            visc_chill = PropsSI('V', 'T', T, 'P', P, fluid_name)
            cond_chill = PropsSI('L', 'T', T, 'P', P, fluid_name)
            
            err_D = abs((rho_chill - rho_cp) / rho_cp)
            err_A = abs((a_chill - a_cp) / a_cp)
            err_V = abs((visc_chill - visc_cp) / visc_cp)
            err_L = abs((cond_chill - cond_cp) / cond_cp)
            
            errors['D'].append(err_D)
            errors['A'].append(err_A)
            errors['V'].append(err_V)
            errors['L'].append(err_L)
            
        except Exception as e:
            print(f"Failed at T={T}, P={P}: {e}")
            
    print(f"  Max Error Density: {max(errors['D']):.2e}")
    print(f"  Max Error Sound Speed: {max(errors['A']):.2e}")
    print(f"  Max Error Viscosity: {max(errors['V']):.2e}")
    print(f"  Max Error Conductivity: {max(errors['L']):.2e}")
    
    if max(errors['D']) > 1e-6: print("  -> Density Mismatch!")
    if max(errors['V']) > 1e-4: print("  -> Viscosity Mismatch!")
    if max(errors['L']) > 1e-4: print("  -> Conductivity Mismatch!")

fluids = [
    "Nitrogen", "Oxygen", "Argon", "CarbonDioxide", 
    "Methane", "Ethane", "Propane", "n-Butane", 
    "IsoButane", "n-Dodecane", "Hydrogen"
]

for f in fluids:
    verify_fluid(f)
