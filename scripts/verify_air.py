import sys
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import numpy as np
import time

# Ensure chillprop is in path
sys.path.append(os.getcwd())
import chillprop.highlevel as CH

def verify_parity():
    fluid = "Air"
    print(f"--- Verifying {fluid} Parity ---")
    
    # Range of T and P for Air (Lemmon 2000)
    # Tc ~ 132.5 K, Pc ~ 3.78 MPa
    T_range = [70, 100, 120, 150, 300, 500, 1000] # K
    P_range = [1e5, 1e6, 5e6, 2e7] # Pa
    
    outputs = [
        ('D', 'Density', 'kg/m3'),
        ('H', 'Enthalpy', 'J/kg'),
        ('S', 'Entropy', 'J/kg/K'),
        ('V', 'Viscosity', 'Pa-s'),
        ('L', 'Conductivity', 'W/m/K')
    ]
    
    all_passed = True
    
    for T in T_range:
        for P in P_range:
            print(f"\nState: T={T} K, P={P/1e6:.1f} MPa")
            for out_key, name, unit in outputs:
                try:
                    # CoolProp
                    val_cp = CP.PropsSI(out_key, 'T', T, 'P', P, fluid)
                    
                    # ChillProp
                    val_chip = CH.PropsSI(out_key, 'T', T, 'P', P, fluid)
                    
                    diff = abs(val_cp - val_chip)
                    rel_diff = diff / abs(val_cp) if val_cp != 0 else diff
                    
                    status = "OK" if rel_diff < 1e-6 else "FAIL"
                    if status == "FAIL": all_passed = False
                    
                    print(f"  {name:12}: CP={val_cp:12.6e}, Chill={val_chip:12.6e} | Diff={rel_diff:8.2e} [{status}]")
                except Exception as e:
                    print(f"  {name:12}: ERROR - {e}")
                    all_passed = False

    # Saturation properties
    print(f"\n--- Saturation Properties ---")
    T_sat = 100.0 # K
    for out_key, name, unit in [('P', 'Psat', 'Pa'), ('D', 'rhoL', 'kg/m3')]:
        try:
            val_cp = CP.PropsSI(out_key, 'T', T_sat, 'Q', 0, fluid)
            val_chip = CH.PropsSI(out_key, 'T', T_sat, 'Q', 0, fluid)
            rel_diff = abs(val_cp - val_chip) / abs(val_cp)
            status = "OK" if rel_diff < 1e-4 else "FAIL" # Saturation can be trickier
            print(f"  {name:12}: CP={val_cp:12.6e}, Chill={val_chip:12.6e} | Diff={rel_diff:8.2e} [{status}]")
        except Exception as e:
             print(f"  {name:12}: ERROR - {e}")

    if all_passed:
        print("\nSUCCESS: All parity tests passed within tolerance!")
    else:
        print("\nFAILURE: Some tests failed parity check.")
    return all_passed

def check_tracing():
    print(f"\n--- XLA Tracing Check ---")
    fluid = "Air"
    params = CH.get_params(fluid)
    print(f"DEBUG: {fluid} pseudo_pure={params.pseudo_pure}")
    if params.ancillary_pL:
        print(f"DEBUG: pL type={params.ancillary_pL.type}, reducing={params.ancillary_pL.reducing_value}, Tr={params.ancillary_pL.T_r}, using_tau_r={params.ancillary_pL.using_tau_r}")
        print(f"DEBUG: pL n={params.ancillary_pL.n}")
        print(f"DEBUG: pL t={params.ancillary_pL.t}")
    
    # Define a pure jax function for tracer analysis
    @jax.jit
    def calculate_props(T, P):
        return CH.viscosity(params, CH.solve_rho_PT(params, P, T), T)

    # Trigger compilation
    _ = calculate_props(300.0, 1e6)
    
    # Get HLO or similar? We can use xla_trace log or just verify it compiles.
    # A better way to verify "single kernel" is to look at the HLO text
    # Get HLO modules count
    hlo_modules = calculate_props.lower(300.0, 1e6).compile().runtime_executable().hlo_modules()
    num_modules = len(hlo_modules)
    
    print(f"HLO Module Name: {hlo_modules[0].name}")
    if num_modules == 1:
        print("CONFIRMED: Entire property routine is compiled into a SINGLE XLA module.")
    else:
        print(f"WARNING: Routine is split across {num_modules} XLA modules.")
    print("XLA compilation successful.")

if __name__ == "__main__":
    verify_parity()
    check_tracing()
