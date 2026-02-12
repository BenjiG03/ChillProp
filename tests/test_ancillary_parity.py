import pytest
import jax
import jax.numpy as jnp
import numpy as np
import CoolProp.CoolProp as CP
from chillprop.parameters import FluidParameters
from chillprop.phases import psat_anc, rhol_anc, rhov_anc
from scripts.extract_params import extract_fluid_params

jax.config.update("jax_enable_x64", True)

@pytest.fixture(scope="module")
def params():
    data = extract_fluid_params('Nitrogen')
    return FluidParameters.from_json(data)

def test_ancillary_parity(params):
    # Test at several subcritical temperatures
    Tc = params.Tc
    T_test = [70, 80, 100, 120]
    
    for T in T_test:
        # ChillProp
        p_c = float(psat_anc(params, jnp.array(float(T))))
        rl_c = float(rhol_anc(params, jnp.array(float(T))))
        rv_c = float(rhov_anc(params, jnp.array(float(T))))
        
        # CoolProp Reference
        # Note: PropsSI('P', 'T', T, 'Q', 0/1, fluid) uses the full HEOS/VLE solver.
        # To test the ANCILLARIES specifically, we might need a different API if available,
        # but usually the ancillaries match PropsSI very closely.
        
        p_ref = CP.PropsSI('P', 'T', T, 'Q', 0, 'Nitrogen')
        rl_ref = CP.PropsSI('Dmolar', 'T', T, 'Q', 0, 'Nitrogen')
        rv_ref = CP.PropsSI('Dmolar', 'T', T, 'Q', 1, 'Nitrogen')
        
        print(f"T={T} K")
        print(f"  P: JAX={p_c:.2f}, CP={p_ref:.2f}, rel_err={abs(p_c-p_ref)/p_ref:.2e}")
        print(f"  rhoL: JAX={rl_c:.2f}, CP={rl_ref:.2f}, rel_err={abs(rl_c-rl_ref)/rl_ref:.2e}")
        print(f"  rhoV: JAX={rv_c:.2f}, CP={rv_ref:.2f}, rel_err={abs(rv_c-rv_ref)/rv_ref:.2e}")
        
        # Ancillaries are typically accurate to 0.1% or better.
        assert np.isclose(p_c, p_ref, rtol=1e-3)
        assert np.isclose(rl_c, rl_ref, rtol=1e-3)
        assert np.isclose(rv_c, rv_ref, rtol=1e-3)
