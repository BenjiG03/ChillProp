import sys
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import CoolProp.CoolProp as CP

sys.path.append(os.getcwd())
import chillprop.highlevel as CH

def analyze():
    fluid = "Air"
    print(f"--- Error Analysis for {fluid} ---")
    
    # Define points
    # Low T: Liquid and Gas below 150K
    T_low = np.linspace(70, 150, 20)
    # High T: Gas/Supercritical above 150K
    T_high = np.linspace(150, 4000, 20)
    
    P_values = [1e5, 1e6, 2e7] 
    
    properties = ['D', 'H', 'S', 'L'] # Density, Enthalpy, Entropy, Conductivity
    
    for prop in properties:
        print(f"\nProperty: {prop}")
        
        for region_name, T_range in [("Low T (<150K)", T_low), ("High T (>150K)", T_high)]:
            errors = []
            for P in P_values:
                for T in T_range:
                    try:
                        val_cp = CP.PropsSI(prop, 'T', T, 'P', P, fluid)
                        val_chill = float(CH.PropsSI(prop, 'T', T, 'P', P, fluid))
                        
                        if val_cp != 0:
                            rel_err = abs(val_chill - val_cp) / abs(val_cp)
                            errors.append(rel_err)
                    except:
                        pass
            
            if errors:
                max_err = np.max(errors)
                rmse = np.sqrt(np.mean(np.array(errors)**2))
                print(f"  {region_name}: Max={max_err:.2e}, RMSE={rmse:.2e}")
            else:
                print(f"  {region_name}: No valid points")

if __name__ == "__main__":
    analyze()
