import numpy as np
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from chillprop.solver import solve_rho_PT
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def check_error(fluid="Nitrogen"):
    print(f"Checking Density Error for {fluid}...")
    T_min = CP.PropsSI("Tmin", fluid)
    T_vals = np.linspace(T_min + 5, 500, 50)
    P_vals = np.logspace(5, 7.5, 50)
    
    params = CH.get_params(fluid)
    
    @jax.jit
    def get_rho(P, T):
        return solve_rho_PT(params, P, T)
    
    max_err = 0.0
    worst_point = (0, 0)
    
    # Grid search
    for P in P_vals:
        for T in T_vals:
            try:
                # CoolProp
                rho_cp = CP.PropsSI("Dmolar", "T", T, "P", P, fluid)
                # ChillProp
                rho_ch = float(get_rho(jnp.array(P), jnp.array(T)))
                
                # Filter out valid results
                if np.isnan(rho_ch) or rho_ch < 0:
                    print(f"Invalid result at T={T:.2f}, P={P:.2e}: {rho_ch}")
                    continue
                    
                err = abs(rho_ch - rho_cp) / rho_cp
                if err > max_err:
                    max_err = err
                    worst_point = (T, P)
            except Exception as e:
                pass
                
    print(f"\nMax Relative Error: {max_err:.2%}")
    print(f"Worst Point: T={worst_point[0]:.2f} K, P={worst_point[1]:.2e} Pa")
    
    if max_err > 0.01: # > 1%
        print("SIGNIFICANT ERROR DETECTED")
        # Check specific point
        T, P = worst_point
        rho_cp = CP.PropsSI("Dmolar", "T", T, "P", P, fluid)
        rho_ch = float(get_rho(jnp.array(P), jnp.array(T)))
        print(f"CP: {rho_cp}, CH: {rho_ch}")

if __name__ == "__main__":
    check_error()
