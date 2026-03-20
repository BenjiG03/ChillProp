import jax
import jax.numpy as jnp
import numpy as np
from chillprop.highlevel import PropsSI

def test_scalar_props():
    rho = PropsSI("D", "T", 300.0, "P", 1e6, "Air")
    assert isinstance(rho, float) or isinstance(rho, jax.Array)
    assert 11.0 < float(rho) < 12.0 # Air ~1.2 kg/m3 at 1 atm, 10 atm -> 12 kg/m3

def test_vector_props_jax():
    T = jnp.array([300.0, 310.0])
    P = jnp.array([1e6, 1e6])
    h = PropsSI("H", "T", T, "P", P, "Air")
    assert h.shape == (2,)

def def_test_vector_props_numpy():
    T = np.array([300.0, 310.0, 320.0])
    P = np.array([1e6, 2e6, 3e6])
    s = PropsSI("S", "T", T, "P", P, "Nitrogen")
    assert s.shape == (3,)

def test_constants():
    R = PropsSI("gas_constant", "T", 300.0, "P", 1e6, "Air")
    M = PropsSI("molar_mass", "T", 300.0, "P", 1e6, "Air")
    # Air molar mass is around 0.02896
    assert 0.028 < float(M) < 0.029
    assert 8.314 < float(R) < 8.315
