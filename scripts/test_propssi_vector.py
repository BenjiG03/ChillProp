import jax.numpy as jnp
import numpy as np
from chillprop.highlevel import PropsSI

def main():
    # 1. Test scalar inputs
    rho_single = PropsSI("D", "T", 300.0, "P", 1e6, "Air")
    print("Scalar Density of Air (300 K, 1 MPa):", rho_single)
    
    # 2. Test vector inputs via JAX array
    T_vec = jnp.array([300.0, 310.0, 320.0])
    P_vec = jnp.array([1e6, 1e6, 1.5e6])
    rho_vec = PropsSI("D", "T", T_vec, "P", P_vec, "Air")
    print("Vector Density (JAX):", rho_vec)
    
    # 3. Test vector inputs via Numpy array
    T_np = np.array([400.0, 500.0])
    P_np = np.array([1e6, 2e6])
    h_np = PropsSI("H", "T", T_np, "P", P_np, "Air")
    print("Vector Enthalpy (NumPy):", h_np)
    
if __name__ == '__main__':
    main()
