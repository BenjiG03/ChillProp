
import jax
import jax.numpy as jnp
from chillprop import parameters, transport
import json
import CoolProp.CoolProp as CP

jax.config.update('jax_enable_x64', True)

def check_fluid(fluid_name, T_test=None, rho_test=None):
    json_path = f'c:/Users/Benji/Documents/ChillProp/CoolProp/dev/fluids/{fluid_name}.json'
    with open(json_path) as f:
        data = json.load(f)
    params = parameters.FluidParameters.from_json(data)
    
    Tc = CP.PropsSI('Tcrit', fluid_name)
    rhoc_molar = CP.PropsSI('rhomolar_critical', fluid_name)
    
    T = T_test if T_test else 1.1 * Tc
    rho = rho_test if rho_test else 0.5 * rhoc_molar
    
    print(f"--- {fluid_name} | T={T:.3f} K | rho={rho:.3f} mol/m3 ---")
    
    # Viscosity
    eta_total = transport.viscosity(params, rho, T)
    eta_cp = CP.PropsSI('viscosity', 'T', T, 'Dmolar', rho, fluid_name)
    print(f"Visc: Total={float(eta_total):.6e}, CP={eta_cp:.6e}, Error={abs(eta_total-eta_cp)/eta_cp:.2%}")
    
    # Conductivity
    lambda_total = transport.thermal_conductivity(params, rho, T)
    lambda_cp = CP.PropsSI('conductivity', 'T', T, 'Dmolar', rho, fluid_name)
    print(f"Cond: Total={float(lambda_total):.6e}, CP={lambda_cp:.6e}, Error={abs(lambda_total-lambda_cp)/lambda_cp:.2%}")

if __name__ == "__main__":
    for f in ["Nitrogen", "Oxygen", "Methane", "CarbonDioxide"]:
        check_fluid(f)
        # Also check far from critical (300K)
        Tc = CP.PropsSI('Tcrit', f)
        if Tc < 270:
            check_fluid(f, T_test=300.0, rho_test=1.0) # dilute-ish gas
