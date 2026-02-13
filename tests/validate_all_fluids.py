import os
import glob
import json
import time
import numpy as np
import CoolProp.CoolProp as CP
from chillprop import highlevel, parameters
import jax
import jax.numpy as jnp

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

def validate_all():
    fluids_dir = r"c:\Users\Benji\Documents\ChillProp\CoolProp\dev\fluids"
    json_files = glob.glob(os.path.join(fluids_dir, "*.json"))
    
    print(f"Found {len(json_files)} fluid files.")
    
    passed = []
    failed_load = []
    failed_calc = []
    
    print(f"{'Fluid':<20} | {'Status':<10} | {'Error'}")
    print("-" * 60)
    
    for json_path in json_files:
        fluid_name = os.path.basename(json_path).replace(".json", "")
        
        # Skip disabled files
        if "_disabled" in fluid_name:
            continue
            
        try:
            # 1. Loading
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            params = parameters.FluidParameters.from_json(data)
            
            # 2. Calculation (Single point check)
            # Use typical state: T = 1.1 Tc, rho = 0.5 rhoc (Supercritical, generally safe)
            Tc = params.Tc
            rhoc = params.rhoc
            
            T_test = 1.1 * Tc
            rho_test = 0.5 * rhoc
            
            # ChillProp
            P_chill = float(highlevel.PropsSI("P", "T", T_test, "D", rho_test, fluid_name))
            
            # CoolProp
            P_cp = CP.PropsSI("P", "T", T_test, "D", rho_test, fluid_name)
            
            error = abs(P_chill - P_cp) / P_cp
            
            if error < 1e-4:
                print(f"{fluid_name:<20} | PASS       | Error: {error:.2e}")
                passed.append(fluid_name)
            else:
                print(f"{fluid_name:<20} | FAIL       | Error: {error:.2e}")
                failed_calc.append((fluid_name, error))
                
        except Exception as e:
            # Shorten error message
            err_msg = str(e).split('\n')[0][:50]
            print(f"{fluid_name:<20} | LOAD FAIL  | {err_msg}")
            failed_load.append((fluid_name, str(e)))

    print("-" * 60)
    print(f"Summary:")
    print(f"Total: {len(json_files)}")
    print(f"Passed: {len(passed)}")
    print(f"Load Failed: {len(failed_load)}")
    print(f"Calc Failed: {len(failed_calc)}")
    
    # Save detailed report
    with open("tests/validation_summary.json", "w") as f:
        json.dump({
            "passed": passed,
            "failed_load": failed_load,
            "failed_calc": failed_calc
        }, f, indent=2)

if __name__ == "__main__":
    validate_all()
