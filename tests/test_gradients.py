import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import CoolProp.CoolProp as CP
from chillprop.parameters import FluidParameters
from chillprop.core import pressure
from scripts.extract_params import extract_fluid_params

jax.config.update("jax_enable_x64", True)

def test_gradient_parity():
    # Load Params
    data = extract_fluid_params('Nitrogen')
    params = FluidParameters.from_json(data)
    
    # Test Point (Supercritical)
    T_vals = [150.0, 300.0]
    rho_vals = [1.0, 1000.0, 10000.0]
    
    jit_dP_drho = eqx.filter_jit(jax.grad(pressure, argnums=1))
    jit_dP_dT = eqx.filter_jit(jax.grad(pressure, argnums=2))
    
    failures = []
    
    print("\nTesting Gradients...")
    for T in T_vals:
        for rho in rho_vals:
            # JAX
            # rho must be molar for core.pressure
            # So dP/drho is Pa / (mol/m^3) = J/mol?
            dp_drho_jax = float(jit_dP_drho(params, jnp.array(rho), jnp.array(T)))
            dp_dt_jax = float(jit_dP_dT(params, jnp.array(rho), jnp.array(T)))
            
            # CoolProp
            # 'd(P)/d(Dmolar)|T'
            try:
                ref_dp_drho = CP.PropsSI('d(P)/d(Dmolar)|T', 'T', T, 'Dmolar', rho, 'Nitrogen')
                ref_dp_dt = CP.PropsSI('d(P)/d(T)|Dmolar', 'T', T, 'Dmolar', rho, 'Nitrogen')
            except Exception as e:
                print(f"Skipping {T}, {rho}: {e}")
                continue
                
            # Compare
            rtol = 1e-5 
            
            if not np.isclose(dp_drho_jax, ref_dp_drho, rtol=rtol):
                failures.append(f"dP/drho mismatch at {T} K, {rho} mol/m3: JAX={dp_drho_jax:.4e}, CP={ref_dp_drho:.4e}")
            
            if not np.isclose(dp_dt_jax, ref_dp_dt, rtol=rtol):
                failures.append(f"dP/dT mismatch at {T} K, {rho} mol/m3: JAX={dp_dt_jax:.4e}, CP={ref_dp_dt:.4e}")
                
            print(f"T={T}, rho={rho} -> dP/drho: JAX={dp_drho_jax:.2e} CP={ref_dp_drho:.2e} | dP/dT: JAX={dp_dt_jax:.2e} CP={ref_dp_dt:.2e}")

    assert not failures, "\n".join(failures)
    print("Gradient Parity Passed!")

if __name__ == "__main__":
    test_gradient_parity()
