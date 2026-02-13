import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from chillprop.highlevel import PropsSI
from chillprop import core, highlevel
import CoolProp.CoolProp as CP

jax.config.update("jax_enable_x64", True)

def check_nitrogen():
    fluid = "Nitrogen"
    T = 300.0
    P = 101325.0
    
    print(f"Checking {fluid} at {T} K, {P} Pa")
    
    # ChillProp params
    params = highlevel.get_params(fluid)
    
    # 1. Density
    rho_cp = CP.PropsSI('D', 'T', T, 'P', P, fluid)
    rho_chill = PropsSI('D', 'T', T, 'P', P, fluid)
    print(f"Density: CP={rho_cp:.6f}, Chill={rho_chill:.6f}, Diff={abs(rho_cp-rho_chill)/rho_cp:.2e}")
    
    # Use ChillProp density for low-level checks to avoid solver diffs
    rho = rho_chill
    
    # 2. Cv
    cv_cp = CP.PropsSI('CVMASS', 'T', T, 'D', rho, fluid)
    # core.cvmolar returns molar, convert to mass
    cv_chill_molar = core.cvmolar(params, rho/params.M, T) # core takes rho_molar
    cv_chill = cv_chill_molar / params.M
    print(f"Cv (mass): CP={cv_cp:.6f}, Chill={cv_chill:.6f}, Diff={abs(cv_cp-cv_chill)/cv_cp:.2e}")
    
    # 3. Cp
    cp_cp = CP.PropsSI('CPMASS', 'T', T, 'D', rho, fluid)
    cp_chill_molar = core.cpmolar(params, rho/params.M, T)
    cp_chill = cp_chill_molar / params.M
    print(f"Cp (mass): CP={cp_cp:.6f}, Chill={cp_chill:.6f}, Diff={abs(cp_cp-cp_chill)/cp_cp:.2e}")
    
    # 4. Speed of Sound
    w_cp = CP.PropsSI('A', 'T', T, 'D', rho, fluid)
    w_chill = core.speed_sound(params, rho/params.M, T)
    print(f"Speed of Sound: CP={w_cp:.6f}, Chill={w_chill:.6f}, Diff={abs(w_cp-w_chill)/w_cp:.2e}")

    # 5. Check Alpha deriv calculations
    try:
        from chillprop.core import get_alpha_and_derivs
        vals = get_alpha_and_derivs(params, rho/params.M, T)
        print("Derivatives:")
        for k, v in vals.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Could not check derivatives: {e}")

if __name__ == "__main__":
    check_nitrogen()
