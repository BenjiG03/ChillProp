
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from chillprop.highlevel import PropsSI
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

OUTPUT_DIR = "docs/plots/combustion"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_plot(fluid_name):
    print(f"Generating plots for {fluid_name}...")
    
    # Get critical point from CoolProp
    try:
        Tc = CP.PropsSI("Tcrit", fluid_name)
        Pc = CP.PropsSI("Pcrit", fluid_name)
        T_min = CP.PropsSI("Tmin", fluid_name)
        T_max = CP.PropsSI("Tmax", fluid_name)
    except Exception as e:
        print(f"Skipping {fluid_name} due to CP error: {e}")
        return

    # Create a T-P grid
    # We want to show parity across a wide range.
    # Let's do isobars.
    pressure_levels = [1e5, 10e5, 50e5, 100e5] # 1, 10, 50, 100 bar
    T_vals = np.linspace(max(T_min, 100), min(T_max, 1000), 50)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{fluid_name} Parity Check (ChillProp vs CoolProp)", fontsize=16)
    
    # Plot 1: Density - Relative Error
    ax = axs[0, 0]
    for P in pressure_levels:
        rho_cp = []
        rho_chill = []
        for T in T_vals:
            try:
                rho_cp.append(CP.PropsSI('D', 'T', T, 'P', P, fluid_name))
                rho_chill.append(PropsSI('D', 'T', T, 'P', P, fluid_name))
            except:
                rho_cp.append(np.nan)
                rho_chill.append(np.nan)
        
        rho_cp = np.array(rho_cp)
        rho_chill = np.array(rho_chill)
        err = np.abs((rho_chill - rho_cp) / rho_cp)
        ax.plot(T_vals, err, label=f"P={P/1e5:.0f} bar")
        
    ax.set_title("Density Relative Error")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("|(Chill - CP) / CP|")
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-")
    ax.legend()
    
    # Plot 2: Speed of Sound - Relative Error
    ax = axs[0, 1]
    for P in pressure_levels:
        w_cp = []
        w_chill = []
        for T in T_vals:
            try:
                w_cp.append(CP.PropsSI('A', 'T', T, 'P', P, fluid_name))
                w_chill.append(PropsSI('A', 'T', T, 'P', P, fluid_name))
            except:
                w_cp.append(np.nan)
                w_chill.append(np.nan)
                
        w_cp = np.array(w_cp)
        w_chill = np.array(w_chill)
        err = np.abs((w_chill - w_cp) / w_cp)
        ax.plot(T_vals, err, label=f"P={P/1e5:.0f} bar")
        
    ax.set_title("Speed of Sound Relative Error")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Rel Error")
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-")

    # Plot 3: Viscosity - Absolute Comparison (Log Scale)
    ax = axs[1, 0]
    for P in pressure_levels:
        v_cp = []
        v_chill = []
        for T in T_vals:
            try:
                v_cp.append(CP.PropsSI('V', 'T', T, 'P', P, fluid_name))
                v_chill.append(PropsSI('V', 'T', T, 'P', P, fluid_name))
            except:
                v_cp.append(np.nan)
                v_chill.append(np.nan)
        
        # Plot lines
        line, = ax.plot(T_vals, v_cp, '-', label=f"CP {P/1e5:.0f}bar")
        ax.plot(T_vals, v_chill, '--', color=line.get_color(), label=f"Chill {P/1e5:.0f}bar")
        
    ax.set_title("Viscosity [Pa-s]")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Viscosity")
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-")
    # ax.legend() # Too cluttered

    # Plot 4: Conductivity - Absolute Comparison
    ax = axs[1, 1]
    for P in pressure_levels:
        l_cp = []
        l_chill = []
        for T in T_vals:
            try:
                l_cp.append(CP.PropsSI('L', 'T', T, 'P', P, fluid_name))
                l_chill.append(PropsSI('L', 'T', T, 'P', P, fluid_name))
            except:
                l_cp.append(np.nan)
                l_chill.append(np.nan)
        
        line, = ax.plot(T_vals, l_cp, '-', label=f"CP {P/1e5:.0f}bar")
        ax.plot(T_vals, l_chill, '--', color=line.get_color(), label=f"Chill")
        
    ax.set_title("Thermal Conductivity [W/m/K]")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Conductivity")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fluid_name}_parity.png")
    plt.close()

fluids = [
    "Nitrogen", "Oxygen", "Argon", "CarbonDioxide", 
    "Methane", "Ethane", "Propane", "Hydrogen"
]

for f in fluids:
    generate_plot(f)
