
import jax
import jax.numpy as jnp
from chillprop import parameters, transport
import json
import CoolProp.CoolProp as CP

jax.config.update('jax_enable_x64', True)

def debug_v(fluid_name):
    json_path = f'c:/Users/Benji/Documents/ChillProp/CoolProp/dev/fluids/{fluid_name}.json'
    with open(json_path) as f:
        data = json.load(f)
    params = parameters.FluidParameters.from_json(data)
    
    Tc = CP.PropsSI('Tcrit', fluid_name)
    rhoc = CP.PropsSI('rhocrit', fluid_name)
    
    T = 1.1 * Tc
    rho = 0.5 * rhoc
    
    # ChillProp parts
    v = params.viscosity
    eta0 = transport.viscosity_dilute(params, T)
    eta_ho = 0.0
    
    # Manually call the ho part to see where it goes
    if isinstance(v.higher_order, parameters.ViscosityFrictionTheory):
        print(f"[{fluid_name}] Logic: Friction Theory")
    elif isinstance(v.higher_order, dict):
        print(f"[{fluid_name}] Logic: Dict (MBH or other)")
    else:
        print(f"[{fluid_name}] Logic: Unknown/None")
        
    eta_total = transport.viscosity(params, rho, T)
    eta_cp = CP.PropsSI('viscosity', 'T', T, 'Dmolar', rho, fluid_name)
    
    print(f"[{fluid_name}] T={T:.3f}, rho={rho:.3f}")
    print(f"[{fluid_name}] eta0={float(eta0):.6e}")
    print(f"[{fluid_name}] total={float(eta_total):.6e}")
    print(f"[{fluid_name}] CP={eta_cp:.6e}")
    print(f"[{fluid_name}] Error={abs(eta_total-eta_cp)/eta_cp:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    for f in ['Ammonia', 'Methane']:
        try:
            debug_v(f)
        except Exception as e:
            print(f"Error for {f}: {e}")
