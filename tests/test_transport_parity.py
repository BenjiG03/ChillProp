import jax
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from chillprop.transport import viscosity, thermal_conductivity
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("fluid", ["Nitrogen"])
@pytest.mark.parametrize("T", [100.0, 300.0, 500.0])
@pytest.mark.parametrize("P", [1e6, 5e6])
def test_transport_parity(fluid, T, P):
    params = CH.get_params(fluid)
    
    # CoolProp
    as_cp = CP.AbstractState("HEOS", fluid)
    as_cp.update(CP.PT_INPUTS, P, T)
    mu_cp = as_cp.viscosity()
    L_cp = as_cp.conductivity()
    rho = as_cp.rhomolar()
    
    # ChillProp
    mu_ch = float(viscosity(params, jnp.array(rho), jnp.array(T)))
    L_ch = float(thermal_conductivity(params, jnp.array(rho), jnp.array(T)))
    
    print(f"\n{fluid} T={T} P={P}")
    print(f"Viscosity: CP={mu_cp:.2e}, CH={mu_ch:.2e}, rel_err={abs(mu_ch-mu_cp)/mu_cp:.2e}")
    print(f"Conductivity: CP={L_cp:.2e}, CH={L_ch:.2e}, rel_err={abs(L_ch-L_cp)/L_cp:.2e}")
    
    assert np.isclose(mu_ch, mu_cp, rtol=1e-3)
    # Conductivity might be worse due to missing critical enhancement
    assert np.isclose(mu_ch, mu_cp, rtol=5e-2)
