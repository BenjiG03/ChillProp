import pytest
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
import numpy as np

def test_PropsSI_parity():
    fluid = "Nitrogen"
    T = 300.0
    P = 1e6
    
    # 1. PT input
    cp_d = CP.PropsSI('D', 'T', T, 'P', P, fluid)
    ch_d = CH.PropsSI('D', 'T', T, 'P', P, fluid)
    assert np.isclose(ch_d, cp_d, rtol=1e-6)
    
    # 2. Ph input
    h = CP.PropsSI('H', 'T', T, 'P', P, fluid)
    ch_t = CH.PropsSI('T', 'P', P, 'H', h, fluid)
    assert np.isclose(ch_t, T, rtol=1e-5)
    
    # 3. PS input
    s = CP.PropsSI('S', 'T', T, 'P', P, fluid)
    ch_t2 = CH.PropsSI('T', 'P', P, 'S', s, fluid)
    assert np.isclose(ch_t2, T, rtol=1e-5)

def test_AbstractState_parity():
    as_cp = CP.AbstractState("HEOS", "Nitrogen")
    as_ch = CH.AbstractState("HEOS", "Nitrogen")
    
    T = 250.0
    P = 2e6
    
    as_cp.update(CP.PT_INPUTS, P, T)
    as_ch.update(CP.PT_INPUTS, P, T)
    
    assert np.isclose(as_ch.rhomolar(), as_cp.rhomolar(), rtol=1e-6)
    assert np.isclose(as_ch.p(), as_cp.p(), rtol=1e-6)
    assert np.isclose(as_ch.hmolar(), as_cp.hmolar(), rtol=1e-6)

def test_PropsSI_quality():
    fluid = "Nitrogen"
    T = 100.0
    Q = 0.5
    
    # Get density at quality 0.5
    rho_ref = CP.PropsSI('Dmolar', 'T', T, 'Q', Q, fluid)
    rho_ch = CH.PropsSI('Dmolar', 'T', T, 'Q', Q, fluid)
    
    assert np.isclose(rho_ch, rho_ref, rtol=1e-5)
    
    # Get H at quality 0.5
    h_ref = CP.PropsSI('Hmolar', 'T', T, 'Q', Q, fluid)
    h_ch = CH.PropsSI('Hmolar', 'T', T, 'Q', Q, fluid)
    assert np.isclose(h_ch, h_ref, rtol=1e-5)
    
    # Get Q back
    q_calc = CH.PropsSI('Q', 'T', T, 'D', CP.PropsSI('D', 'T', T, 'Q', Q, fluid), fluid)
    assert np.isclose(q_calc, Q, rtol=1e-4)


