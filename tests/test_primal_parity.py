import pytest
import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
import CoolProp.CoolProp as CP
from chillprop.highlevel import get_params
from chillprop.core import pressure, enthalpy, entropy

# Enable double precision
jax.config.update("jax_enable_x64", True)

@pytest.fixture(scope="module")
def nitrogen_params():
    return get_params("Nitrogen")

def test_primal_parity_grid(nitrogen_params):
    fluid = 'Nitrogen'
    Tc = nitrogen_params.Tc
    rhoc = nitrogen_params.rhoc
    
    # Grid definition - Start with supercritical to verify EOS kernels
    T_factors = [1.2, 1.5, 2.0]
    rho_factors = [0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5]
    
    # Also include saturation densities at subcritical temperatures
    # For T < Tc
    extra_points = []
    # for tf in [0.5, 0.8, 0.99]:
    #     T = tf * Tc
    #     try:
    #         rho_liq = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)
    #         rho_vap = CP.PropsSI('D', 'T', T, 'Q', 1, fluid)
    #         extra_points.append((T, rho_liq))
    #         extra_points.append((T, rho_vap))
    #         extra_points.append((T, rho_liq * 1.001))
    #         extra_points.append((T, rho_vap * 0.999))
    #     except:
    #         pass # Skip if fails
            
    # Combine
    test_points = []
    for tf in T_factors:
        for rf in rho_factors:
            test_points.append((tf * Tc, rf * rhoc))
    
    test_points.extend(extra_points)
    
    # Run tests
    failures = []
    
    params = nitrogen_params
    
    # We can vet all points in a JIT-compiled loop or vmap?
    # The prompt asks for verification.
    # Let's verify point by point for clear error reporting, but use JIT for speed.
    
    # JIT compiled functions
    jit_p = eqx.filter_jit(pressure)
    jit_h = eqx.filter_jit(enthalpy)
    jit_s = eqx.filter_jit(entropy)
    
    # Warm up
    jit_p(params, jnp.array(300.0), jnp.array(10.0))
    
    print(f"\nTesting {len(test_points)} points...")
    
    for T_val, rho_val in test_points:
        # Inputs must be arrays for JAX? float is fine, converted to array.
        T_jax = jnp.array(T_val)
        rho_jax = jnp.array(rho_val)
        
        # CoolProp Reference
        try:
            ref_p = CP.PropsSI('P', 'T', T_val, 'D', rho_val, fluid)
            ref_h = CP.PropsSI('H', 'T', T_val, 'D', rho_val, fluid) # J/kg
            ref_s = CP.PropsSI('S', 'T', T_val, 'D', rho_val, fluid) # J/kg/K
            
            # Unit conversion?
            # CoolProp PropsSI returns mass-specific units (J/kg) by default? 
            # Or Molar?
            # Standard PropsSI is mass based (SI).
            # Parameters R is J/mol/K.
            # My Code calculates Molar properties?
            # R is usually molar gas constant.
            # Equations: P = rho * R * T ...
            # If rho is molar density (mol/m^3), then P is Pa.
            # If rho is mass density (kg/m^3), and R is specific gas constant (J/kg/K)...
            # In `parameters.py`, R is `eos['gas_constant']`. 8.314...
            # This is MOLAR gas constant.
            # So `rho` input to my functions should be MOLAR density.
            # CoolProp `D` input to PropsSI is MASS density (kg/m^3) or MOLAR (mol/m^3)?
            # PropsSI('D') is MASS density (kg/m^3).
            # PropsSI('Dmolar') is MOLAR density.
            
            # I must ensure unit consistency!
            # My `heos.py` uses `delta = rho / rhoc`. `rhoc` from JSON.
            # JSON: "rhomolar_critical": 11183.9. Unit: mol/m^3.
            # So my code expects MOLAR DENSITY.
            
            # So test input `rho_val` in the loop:
            # `rho_factors` * `rhoc` (molar).
            # So `rho_val` is MOLAR density.
            
            # CoolProp PropsSI inputs:
            # If I pass 'D', it expects kg/m^3.
            # If I pass 'Dmolar', it expects mol/m^3.
            # I should use 'Dmolar' for inputting rho_val.
            
            # Ref outputs:
            # 'P': Pa.
            # 'H': J/kg (mass specific).
            # 'S': J/kg/K.
            # 'Hmolar': J/mol.
            # 'Smolar': J/mol/K.
            
            # I should use molar units for validation to avoid Molar Mass conversion issues (though M matches).
            
            ref_p = CP.PropsSI('P', 'T', T_val, 'Dmolar', rho_val, fluid)
            ref_h = CP.PropsSI('Hmolar', 'T', T_val, 'Dmolar', rho_val, fluid)
            ref_s = CP.PropsSI('Smolar', 'T', T_val, 'Dmolar', rho_val, fluid)
            
        except Exception as e:
            # Skip invalid points (e.g. negative pressure in unstable region?)
            # CoolProp might throw error.
            # print(f"Skipping {T_val}, {rho_val}: {e}")
            continue

        # JAX Calc
        calc_p = float(jit_p(params, rho_jax, T_jax))
        calc_h = float(jit_h(params, rho_jax, T_jax))
        calc_s = float(jit_s(params, rho_jax, T_jax))
        
        # Validation
        # Tolerances from prompt: 
        # FP64: 1e-7 relative.
        rtol = 1e-7
        atol = 1e-9 # For near zero values
        
        if not np.allclose(calc_p, ref_p, rtol=rtol, atol=atol):
            failures.append(f"P mismatch at T={T_val:.4f}, rho={rho_val:.4f}: JAX={calc_p:.6e}, CP={ref_p:.6e}, rel_err={abs(calc_p-ref_p)/ref_p:.2e}")
        
        if not np.allclose(calc_h, ref_h, rtol=rtol, atol=atol):
             failures.append(f"H mismatch at T={T_val:.4f}, rho={rho_val:.4f}: JAX={calc_h:.6e}, CP={ref_h:.6e}, rel_err={abs(calc_h-ref_h)/abs(ref_h):.2e}")
             
        if not np.allclose(calc_s, ref_s, rtol=rtol, atol=atol):
             failures.append(f"S mismatch at T={T_val:.4f}, rho={rho_val:.4f}: JAX={calc_s:.6e}, CP={ref_s:.6e}, rel_err={abs(calc_s-ref_s)/abs(ref_s):.2e}")

    assert not failures, "\n".join(failures[:20]) # Limit output

if __name__ == "__main__":
    params = get_params("Nitrogen")
    test_primal_parity_grid(params)
    print("Test passed manually!")
