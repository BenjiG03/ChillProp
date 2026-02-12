import jax
import jax.numpy as jnp
from chillprop.parameters import FluidParameters
from chillprop.phases import solve_vle, rhol_anc, rhov_anc
from scripts.extract_params import extract_fluid_params

jax.config.update("jax_enable_x64", True)

def debug_vle():
    print("Extracting params...")
    data = extract_fluid_params('Nitrogen')
    print("Parsing params...")
    params = FluidParameters.from_json(data)
    
    T = 100.0
    print(f"Testing T={T}")
    
    print("Evaluating Ancillary...")
    rl_a = rhol_anc(params, jnp.array(T))
    rv_a = rhov_anc(params, jnp.array(T))
    print(f"Ancillary: rl={rl_a}, rv={rv_a}")
    
    print("Solving VLE...")
    rho_l, rho_v = solve_vle(params, jnp.array(T))
    print(f"Solved: rho_l={rho_l}, rho_v={rho_v}")


if __name__ == "__main__":
    debug_vle()
