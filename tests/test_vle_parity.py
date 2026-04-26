import pytest
import jax
import jax.numpy as jnp
import numpy as np
import CoolProp.CoolProp as CP
from chillprop.highlevel import get_params
from chillprop.phases import solve_vle
from chillprop.core import pressure

jax.config.update("jax_enable_x64", True)

@pytest.fixture(scope="module")
def params():
    return get_params("Nitrogen")

def test_vle_parity(params):
    # Test at several subcritical temperatures
    T_test = [70.0, 80.0, 100.0, 120.0]
    
    for T in T_test:
        # ChillProp Solver
        rho_l, rho_v = solve_vle(params, jnp.array(float(T)))
        psat_calc = float(pressure(params, rho_l, T))
        
        # CoolProp Reference
        psat_ref = CP.PropsSI('P', 'T', T, 'Q', 0, 'Nitrogen')
        rhol_ref = CP.PropsSI('Dmolar', 'T', T, 'Q', 0, 'Nitrogen')
        rhov_ref = CP.PropsSI('Dmolar', 'T', T, 'Q', 1, 'Nitrogen')
        
        print(f"T={T} K")
        print(f"  P_sat: JAX={psat_calc:.2e}, CP={psat_ref:.2e}")
        print(f"  rhoL: JAX={rho_l:.4f}, CP={rhol_ref:.4f}")
        print(f"  rhoV: JAX={rho_v:.4f}, CP={rhov_ref:.4f}")
        
        assert np.isclose(psat_calc, psat_ref, rtol=1e-6)
        assert np.isclose(float(rho_l), rhol_ref, rtol=1e-6)
        assert np.isclose(float(rho_v), rhov_ref, rtol=1e-6)
