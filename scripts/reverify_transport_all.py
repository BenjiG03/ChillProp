
import sys
import os
sys.path.append(os.getcwd())

import jax
import numpy as np
import CoolProp.CoolProp as CP
from chillprop.highlevel import PropsSI
import pandas as pd

jax.config.update("jax_enable_x64", True)

def check_fluid(fluid):
    print(f"\n--- Checking {fluid} ---")
    
    try:
        T_crit = CP.PropsSI("Tcrit", fluid)
        P_crit = CP.PropsSI("Pcrit", fluid)
        T_min = CP.PropsSI("Tmin", fluid)
        T_max = CP.PropsSI("Tmax", fluid)
    except:
        # Air might fail Tcrit call if not handled right in CP? No, Air has Tcrit.
        # But let's handle just in case.
        print(f"Could not get critical props for {fluid}")
        return

    # Generate points: Gas, Liquid, Supercritical
    # 1. Low P gas
    # 2. High P supercritical
    # 3. Liquid (Subcooled)
    
    test_points = [
        (T_min + 50, 1e5),          # Gas
        (T_crit + 50, P_crit*1.5),  # Supercritical
        (T_min + 20, 10e5),         # Liquid (if P > Psat)
    ]
    
    # Check saturation conditions too? 
    # Transport often tricky there. Let's stick to single phase first.
    
    results = []
    
    for T, P in test_points:
        try:
            # CoolProp values
            rho_cp = CP.PropsSI('D', 'T', T, 'P', P, fluid)
            visc_cp = CP.PropsSI('V', 'T', T, 'P', P, fluid)
            cond_cp = CP.PropsSI('L', 'T', T, 'P', P, fluid)
            
            # ChillProp values
            rho_chill = PropsSI('D', 'T', T, 'P', P, fluid)
            visc_chill = PropsSI('V', 'T', T, 'P', P, fluid)
            cond_chill = PropsSI('L', 'T', T, 'P', P, fluid)
            
            # Errors
            err_rho = abs(rho_chill - rho_cp) / rho_cp
            err_visc = abs(visc_chill - visc_cp) / visc_cp
            err_cond = abs(cond_chill - cond_cp) / cond_cp
            
            results.append({
                'T': T, 'P': P,
                'Rho_CP': rho_cp, 'Rho_Chill': rho_chill, 'Err_Rho': err_rho,
                'Visc_CP': visc_cp, 'Visc_Chill': visc_chill, 'Err_Visc': err_visc,
                'Cond_CP': cond_cp, 'Cond_Chill': cond_chill, 'Err_Cond': err_cond
            })
            
        except Exception as e:
           print(f"Failed at T={T:.1f}, P={P:.1f}: {e}")

    # Summary
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    print("Maximum Errors:")
    print(f"  Density:      {df['Err_Rho'].max():.2e}")
    print(f"  Viscosity:    {df['Err_Visc'].max():.2e}")
    print(f"  Conductivity: {df['Err_Cond'].max():.2e}")
    
    # Filter for high errors
    high_err = df[(df['Err_Visc'] > 0.05) | (df['Err_Cond'] > 0.05)]
    if not high_err.empty:
        print("Large Discrepancies (>5%):")
        for _, row in high_err.iterrows():
            print(f"  T={row['T']:.1f}, P={row['P']:.1e} | "
                  f"V_CP={row['Visc_CP']:.2e}, V_Ch={row['Visc_Chill']:.2e} (E={row['Err_Visc']:.2e}) | "
                  f"L_CP={row['Cond_CP']:.2e}, L_Ch={row['Cond_Chill']:.2e} (E={row['Err_Cond']:.2e})")
    else:
        print("All transport errors < 5%.")

fluids = ["Nitrogen", "Oxygen", "Air", "CarbonDioxide", "Hydrogen", "Methane", "Ethane", "Propane"]
for f in fluids:
    check_fluid(f)
