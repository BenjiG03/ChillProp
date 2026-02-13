import sys
import os
import jax
import traceback

# Add root to path
sys.path.append(os.getcwd())

from chillprop.highlevel import PropsSI

def check_fluids():
    fluids = [
        "Nitrogen", # Baseline (should pass)
        "Oxygen",
        "Argon",
        "CarbonDioxide",
        "Water",
        "Methane",
        "Ethane",
        "Propane",
        "n-Butane",
        "IsoButane",
        "n-Dodecane",
        "Hydrogen"
    ]
    
    results = {}
    
    print(f"{'Fluid':<15} | {'Status':<10} | {'Error'}")
    print("-" * 60)
    
    for fluid in fluids:
        try:
            # Try a simple calculation
            # T = 300 K, P = 1 atm
            rho = PropsSI("D", "T", 300, "P", 101325, fluid)
            results[fluid] = "OK"
            print(f"{fluid:<15} | {'OK':<10} | -")
            
            # Try transport if possible (might fail even if EOS works)
            try:
                visc = PropsSI("V", "T", 300, "P", 101325, fluid)
            except Exception as e:
                 print(f"{fluid:<15} | {'Partial':<10} | Viscosity failed: {str(e)}")

        except Exception as e:
            err_msg = str(e).split('\n')[0] # First line of error
            results[fluid] = "FAIL"
            print(f"{fluid:<15} | {'FAIL':<10} | {err_msg}")

if __name__ == "__main__":
    check_fluids()
