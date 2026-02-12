import jax
import jax.numpy as jnp
import numpy as np
from chillprop.parameters import FluidParameters, ResidualHelmholtzPower
from chillprop.heos import alphar_power, alphar, alpha0
from chillprop.core import pressure
from scripts.extract_params import extract_fluid_params

jax.config.update("jax_enable_x64", True)

def debug_jax():
    # Load Params
    data = extract_fluid_params('Nitrogen')
    params = FluidParameters.from_json(data)
    
    T = 151.4304
    rho = 11183.9015
    
    tau = params.Tc / T
    delta = rho / params.rhoc
    
    print(f"tau={tau}, delta={delta}")
    
    # Check alphar calculation
    forterm = None
    pow_terms = [t for t in params.alphar if isinstance(t, ResidualHelmholtzPower)]
    if pow_terms:
        term = pow_terms[0]
        # Inspect manual calc
        print("L array:", term.l)
        
        # JAX Calc
        val = alphar_power(term, jnp.array(tau), jnp.array(delta))
        print(f"JAX alphar_power val: {val}")
        
        # Manual Check of exp logic
        l = term.l
        exp_factor = jnp.where(l!=0, jnp.exp(-(delta**l)), 1.0)
        print(f"Exp factors: {exp_factor}")
        
        # Check grad
        def func(d):
            return alphar_power(term, jnp.array(tau), d)
        
        grad_val = jax.grad(func)(jnp.array(delta))
        print(f"JAX grad w.r.t delta: {grad_val}")
        
        # Expected grad contribution roughly:
        # Z - 1 = delta * grad
        # debug_heos said Z_contrib = -0.455.
        # So grad should be -0.455 / delta = -0.455.
        print(f"Expected grad: -0.4553")
    
    # Full Pressure with JIT
    import equinox as eqx
    jit_p = eqx.filter_jit(pressure)
    p_val = jit_p(params, jnp.array(rho), jnp.array(T))
    print(f"JAX Pressure (JIT): {p_val}")

if __name__ == "__main__":
    debug_jax()
