import jax
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
import numpy as np

jax.config.update("jax_enable_x64", True)

def test():
    print("AS ref init...")
    as_cp = CP.AbstractState("HEOS", "Nitrogen")
    print("AS chill init...")
    as_ch = CH.AbstractState("HEOS", "Nitrogen")
    
    T = 250.0
    P = 2e6
    
    print("AS ref update...")
    as_cp.update(CP.PT_INPUTS, P, T)
    print("AS chill update...")
    as_ch.update(CP.PT_INPUTS, P, T)
    print("Done update.")
    
    print(f"Ref rho: {as_cp.rhomolar()}")
    print(f"JAX rho: {as_ch.rhomolar()}")

if __name__ == "__main__":
    test()
