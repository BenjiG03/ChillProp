import pytest
import jax
import jax.numpy as jnp
import numpy as np
import CoolProp.CoolProp as CP
from chillprop.parameters import FluidParameters
from chillprop.solver import solve_rho_PT, solve_rhoT_Ph, solve_rhoT_Ps
from scripts.extract_params import extract_fluid_params

jax.config.update("jax_enable_x64", True)

@pytest.fixture(scope="module")
def params():
    data = extract_fluid_params('Nitrogen')
    return FluidParameters.from_json(data)

def test_rho_PT_solver(params):
    T_target = 300.0
    P_target = 1e6 # 1 MPa
    rho_ref = CP.PropsSI('Dmolar', 'T', T_target, 'P', P_target, 'Nitrogen')
    rho_calc = solve_rho_PT(params, jnp.array(P_target), jnp.array(T_target))
    assert np.isclose(float(rho_calc), rho_ref, rtol=1e-7)

def test_rhoT_Ph(params):
    P_target = 1e6
    T_known = 400.0
    h_target = CP.PropsSI('Hmolar', 'T', T_known, 'P', P_target, 'Nitrogen')
    rho_ref = CP.PropsSI('Dmolar', 'T', T_known, 'P', P_target, 'Nitrogen')
    res = solve_rhoT_Ph(params, jnp.array(P_target), jnp.array(h_target))
    rho_calc, T_calc = float(res[0]), float(res[1])
    assert np.isclose(rho_calc, rho_ref, rtol=1e-6)
    assert np.isclose(T_calc, T_known, rtol=1e-6)

def test_rhoT_Ps(params):
    P_target = 2e6
    T_known = 500.0
    s_target = CP.PropsSI('Smolar', 'T', T_known, 'P', P_target, 'Nitrogen')
    rho_ref = CP.PropsSI('Dmolar', 'T', T_known, 'P', P_target, 'Nitrogen')
    res = solve_rhoT_Ps(params, jnp.array(P_target), jnp.array(s_target))
    rho_calc, T_calc = float(res[0]), float(res[1])
    assert np.isclose(rho_calc, rho_ref, rtol=1e-6)
    assert np.isclose(T_calc, T_known, rtol=1e-6)

def test_solver_gradients(params):
    T_target = 300.0
    P_target = 1e6
    
    # 1. d(rho)/d(P)|T
    drho_dp_jax = jax.grad(solve_rho_PT, argnums=1)(params, jnp.array(P_target), jnp.array(T_target))
    
    AS = CP.AbstractState("HEOS", "Nitrogen")
    AS.update(CP.PT_INPUTS, P_target, T_target)
    ref_dp_drho = AS.first_partial_deriv(CP.iP, CP.iDmolar, CP.iT)
    ref_drho_dp = 1.0 / ref_dp_drho
    
    print(f"drho/dP: JAX={drho_dp_jax}, CP={ref_drho_dp}")
    assert np.isclose(float(drho_dp_jax), ref_drho_dp, rtol=1e-5)
    
    # 2. d(T)/d(P)|h (Joule-Thomson derivative-ish)
    def get_T_from_Ph(p, h):
        return solve_rhoT_Ph(params, p, h)[1]
        
    h_target = AS.hmolar()
    dT_dp_h_jax = jax.grad(get_T_from_Ph, argnums=0)(jnp.array(P_target), jnp.array(h_target))
    
    # Ref: (dT/dP)|h = - (dh/dP)|T / (dh/dT)|P
    ref_dh_dp_T = AS.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT)
    ref_dh_dt_P = AS.cpmolar()
    ref_dt_dp_h = - ref_dh_dp_T / ref_dh_dt_P
    
    print(f"dT/dP|h: JAX={dT_dp_h_jax}, CP={ref_dt_dp_h}")
    assert np.isclose(float(dT_dp_h_jax), ref_dt_dp_h, rtol=1e-4) # Slightly looser due to iteration count
