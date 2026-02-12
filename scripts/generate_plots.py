import matplotlib.pyplot as plt
import numpy as np
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from chillprop.solver import solve_rho_PT
from chillprop.phases import solve_vle
from chillprop.transport import viscosity, thermal_conductivity
import jax
import jax.numpy as jnp
import os

jax.config.update("jax_enable_x64", True)

OUTPUT_DIR = "validation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_density_parity(fluid="Nitrogen"):
    print(f"Generating Density Parity Plot for {fluid}...")
    T_min = CP.PropsSI("Tmin", fluid)
    T_crit = CP.PropsSI("Tcrit", fluid)
    P_crit = CP.PropsSI("Pcrit", fluid)
    
    T_vals = np.linspace(T_min + 5, 500, 50)
    P_vals = np.logspace(5, 7.5, 50)
    
    rho_cp = []
    rho_ch = []
    
    params = CH.get_params(fluid)
    
    # Vectorized compute would be faster, but let's just loop for plotting script simplicity
    # or actually use vmap for speed
    
    @jax.jit
    def get_rho(P, T):
        return solve_rho_PT(params, P, T)
        
    for P in P_vals:
        for T in T_vals:
            # Skip near-critical/two-phase for simple parity plot to avoid solver failure noise
            # or just try/except
            try:
                # CoolProp
                r_cp = CP.PropsSI("Dmolar", "T", T, "P", P, fluid)
                # ChillProp
                r_ch = float(get_rho(jnp.array(P), jnp.array(T)))
                
                rho_cp.append(r_cp)
                rho_ch.append(r_ch)
            except:
                pass
                
    plt.figure(figsize=(6, 6))
    plt.scatter(rho_cp, rho_ch, s=1, alpha=0.5)
    plt.plot([min(rho_cp), max(rho_cp)], [min(rho_cp), max(rho_cp)], 'r--')
    plt.xlabel("CoolProp Density [mol/m3]")
    plt.ylabel("ChillProp Density [mol/m3]")
    plt.title(f"{fluid} Density Parity")
    plt.loglog()
    plt.savefig(f"{OUTPUT_DIR}/density_parity_{fluid}.png")
    plt.close()

def plot_vle_envelope(fluid="Nitrogen"):
    print(f"Generating VLE Envelope for {fluid}...")
    T_crit = CP.PropsSI("Tcrit", fluid)
    T_min = CP.PropsSI("Tmin", fluid)
    
    T_vals = np.linspace(T_min, T_crit - 0.1, 100)
    
    rho_l_cp = []
    rho_v_cp = []
    rho_l_ch = []
    rho_v_ch = []
    
    params = CH.get_params(fluid)
    
    @jax.jit
    def get_vle(T):
        return solve_vle(params, T)
    
    for T in T_vals:
        try:
            # CoolProp
            rho_l_cp.append(CP.PropsSI("Dmolar", "T", T, "Q", 0, fluid))
            rho_v_cp.append(CP.PropsSI("Dmolar", "T", T, "Q", 1, fluid))
            
            # ChillProp
            rl, rv = get_vle(jnp.array(T))
            rho_l_ch.append(float(rl))
            rho_v_ch.append(float(rv))
        except:
            pass
            
    plt.figure(figsize=(8, 6))
    plt.plot(rho_l_cp, T_vals, 'k-', label="CoolProp")
    plt.plot(rho_v_cp, T_vals, 'k-')
    plt.plot(rho_l_ch, T_vals, 'r--', label="ChillProp")
    plt.plot(rho_v_ch, T_vals, 'r--')
    plt.xlabel("Density [mol/m3]")
    plt.ylabel("Temperature [K]")
    plt.title(f"{fluid} VLE Envelope")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/vle_envelope_{fluid}.png")
    plt.close()

def plot_transport(fluid="Nitrogen"):
    print(f"Generating Transport Plots for {fluid}...")
    T_vals = np.linspace(100, 500, 100)
    P = 1e6 # 10 bar
    
    mu_cp = []
    mu_ch = []
    k_cp = []
    k_ch = []
    
    params = CH.get_params(fluid)
    
    for T in T_vals:
        # CP
        as_cp = CP.AbstractState("HEOS", fluid)
        as_cp.update(CP.PT_INPUTS, P, T)
        mu_cp.append(as_cp.viscosity())
        k_cp.append(as_cp.conductivity())
        rho = as_cp.rhomolar()
        
        # CH
        mu_ch.append(float(viscosity(params, jnp.array(rho), jnp.array(T))))
        k_ch.append(float(thermal_conductivity(params, jnp.array(rho), jnp.array(T))))
        
    # Viscosity
    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, mu_cp, 'k-', label="CoolProp")
    plt.plot(T_vals, mu_ch, 'r--', label="ChillProp")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Viscosity [Pa-s]")
    plt.title(f"{fluid} Viscosity at 10 bar")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/viscosity_{fluid}.png")
    plt.close()
    
    # Conductivity
    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, k_cp, 'k-', label="CoolProp")
    plt.plot(T_vals, k_ch, 'r--', label="ChillProp")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Thermal Conductivity [W/m/K]")
    plt.title(f"{fluid} Conductivity at 10 bar")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/conductivity_{fluid}.png")
    plt.close()

if __name__ == "__main__":
    plot_density_parity()
    plot_vle_envelope()
    plot_transport()
    print("All plots generated.")
