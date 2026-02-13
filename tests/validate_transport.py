
import os
import glob
import json
import time
import numpy as np
import CoolProp.CoolProp as CP
import jax
jax.config.update('jax_enable_x64', True)
from chillprop import highlevel, parameters, transport

def validate_transport():
    fluids_dir = r"c:\Users\Benji\Documents\ChillProp\CoolProp\dev\fluids"
    json_files = glob.glob(os.path.join(fluids_dir, "*.json"))
    
    print(f"Found {len(json_files)} fluid files.")
    
    test_fluids = [
        "HydrogenSulfide", "n-Pentane", "Ammonia", "Methane", 
        "Nitrogen", "Oxygen", "Ethane", "Propane", 
        "CarbonDioxide", "Hydrogen"
    ]
    
    print(f"{'Fluid':<20} | {'Viscosity':<12} | {'Conductivity':<12}")
    print("-" * 60)
    
    visc_passed = 0
    cond_passed = 0
    total = 0
    
    for fluid_name in test_fluids:
        json_path = os.path.join(fluids_dir, f"{fluid_name}.json")
        if not os.path.exists(json_path):
            continue
            
        try:
            # Get critical props from CoolProp (Molar)
            Tc = CP.PropsSI("Tcrit", fluid_name)
            rhoc = CP.PropsSI("rhomolar_critical", fluid_name)
            
            T_test = 1.1 * Tc
            rho_test = 0.5 * rhoc
            
            # ChillProp Calculation
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                params = parameters.FluidParameters.from_json(data)
                
                # Check Viscosity
                res_visc = "NaN"
                try:
                    eta_cp = CP.PropsSI("viscosity", "T", T_test, "Dmolar", rho_test, fluid_name)
                    eta_chill = float(transport.viscosity(params, rho_test, T_test))
                    if not np.isnan(eta_chill):
                        err_visc = abs(eta_chill - eta_cp) / eta_cp
                        if err_visc < 1e-2:
                            res_visc = "PASS"
                            visc_passed += 1
                        else:
                            res_visc = f"{err_visc:.1%}"
                except ValueError as ve:
                    if "model is not available" in str(ve):
                        res_visc = "N/A"
                    else:
                        res_visc = "ERR_CP"

                # Check Conductivity
                res_cond = "NaN"
                try:
                    cond_cp = CP.PropsSI("conductivity", "T", T_test, "Dmolar", rho_test, fluid_name)
                    cond_chill = float(transport.thermal_conductivity(params, rho_test, T_test))
                    if not np.isnan(cond_chill):
                        err_cond = abs(cond_chill - cond_cp) / cond_cp
                        if err_cond < 1e-2:
                            res_cond = "PASS"
                            cond_passed += 1
                        else:
                            res_cond = f"{err_cond:.1%}"
                except ValueError as ve:
                    if "model is not available" in str(ve):
                        res_cond = "N/A"
                    else:
                        res_cond = "ERR_CP"
                        
            except Exception as e:
                res_visc = "ERR"
                res_cond = "ERR"
            
            print(f"{fluid_name:<20} | {res_visc:<12} | {res_cond:<12}")
            total += 1
            
        except Exception as e:
            pass

    print("-" * 60)
    print(f"Summary:")
    print(f"Total Tested: {total}")
    print(f"Viscosity Passed: {visc_passed}")
    print(f"Conductivity Passed: {cond_passed}")

if __name__ == "__main__":
    validate_transport()
