import numpy as np
import jax
import jax.numpy as jnp
from chillprop.parameters import FluidParameters
from chillprop.solver import solve_rho_PT
from chillprop.core import props
from chillprop.transport import viscosity, thermal_conductivity
import CoolProp.CoolProp as CP
import json
import matplotlib.pyplot as plt
import os
import time

def validate_air():
    # Load chillprop Air
    with open('chillprop/data/Air.json', 'r') as f:
        data = json.load(f)
    params = FluidParameters.from_json(data)
    
    # Generate test grid
    # 70 K to 4000 K, up to 20 MPa. Do a log sweep for Pressure?
    T_range = np.linspace(70, 4000, 50)
    P_range = np.logspace(jnp.log10(1e5), jnp.log10(20e6), 20)
    T_mesh, P_mesh = np.meshgrid(T_range, P_range)
    
    T_flat = T_mesh.flatten()
    P_flat = P_mesh.flatten()
    
    N = len(T_flat)
    
    cp_rho = np.zeros(N)
    cp_h = np.zeros(N)
    cp_s = np.zeros(N)
    cp_eta = np.zeros(N)
    cp_cond = np.zeros(N)
    
    # Evaluate CoolProp
    print("Evaluating CoolProp...")
    for i in range(N):
        try:
            st = CP.AbstractState("HEOS", "Air")
            st.update(CP.PT_INPUTS, P_flat[i], T_flat[i])
            cp_rho[i] = st.rhomass()
            cp_h[i] = st.hmass()
            cp_s[i] = st.smass()
            cp_eta[i] = st.viscosity()
            cp_cond[i] = st.conductivity()
        except Exception as e:
            cp_rho[i] = np.nan
            cp_h[i] = np.nan
            cp_s[i] = np.nan
            cp_eta[i] = np.nan
            cp_cond[i] = np.nan

    import equinox as eqx
    print("Evaluating ChillProp...")
    # ChillProp batch evaluation
    # compile solve_rho
    solve_vmap = eqx.filter_jit(jax.vmap(solve_rho_PT, in_axes=(None, 0, 0)))
    
    t0 = time.time()
    rho_molar = solve_vmap(params, jnp.array(P_flat), jnp.array(T_flat))
    
    # Get properties
    props_vmap = eqx.filter_jit(jax.vmap(props, in_axes=(None, 0, 0)))
    p_dict = props_vmap(params, rho_molar, jnp.array(T_flat))
    
    # Get Transport
    visc_vmap = eqx.filter_jit(jax.vmap(viscosity, in_axes=(None, 0, 0)))
    cond_vmap = eqx.filter_jit(jax.vmap(thermal_conductivity, in_axes=(None, 0, 0)))
    
    eta_molar = visc_vmap(params, rho_molar, jnp.array(T_flat))
    cond_molar = cond_vmap(params, rho_molar, jnp.array(T_flat))
    
    t1 = time.time()
    
    print(f"JAX evaluation took {t1 - t0:.3f} s for {N} points")
    
    # Convert to mass basis
    ch_rho = np.array(rho_molar) * params.M
    ch_h = np.array(p_dict['h']) / params.M
    ch_s = np.array(p_dict['s']) / params.M
    ch_eta = np.array(eta_molar)
    ch_cond = np.array(cond_molar)
    
    # Filter NaNs
    valid = ~np.isnan(cp_rho) & ~np.isnan(ch_rho) & (ch_rho > 0)
    
    os.makedirs('docs/assets', exist_ok=True)
    
    def parity_plot(cp_vals, ch_vals, mask, name, unit):
        plt.figure(figsize=(6, 6))
        plt.plot(cp_vals[mask], ch_vals[mask], '.', markersize=2, alpha=0.5)
        
        mx = np.nanmax(cp_vals[mask])
        mn = np.nanmin(cp_vals[mask])
        if mn > 0:
            plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
        
        plt.xlabel(f'CoolProp {name} [{unit}]')
        plt.ylabel(f'ChillProp {name} [{unit}]')
        plt.title(f'{name} Parity (Air)')
        plt.tight_layout()
        plt.savefig(f'docs/assets/{name}_parity.png', dpi=300)
        plt.close()
        
        # calculate max relative error
        rel_err = np.abs(cp_vals[mask] - ch_vals[mask]) / np.maximum(np.abs(cp_vals[mask]), 1e-10)
        max_err = np.nanmax(rel_err)
        print(f"Max relative error for {name}: {max_err:.2e}")
        return max_err
        
    parity_plot(cp_rho, ch_rho, valid, 'Density', 'kg/m^3')
    parity_plot(cp_h, ch_h, valid, 'Enthalpy', 'J/kg')
    parity_plot(cp_s, ch_s, valid, 'Entropy', 'J/kg/K')
    
    valid_trans = valid & ~np.isnan(cp_eta) & ~np.isnan(ch_eta)
    parity_plot(cp_eta, ch_eta, valid_trans, 'Viscosity', 'Pa-s')
    parity_plot(cp_cond, ch_cond, valid_trans, 'Conductivity', 'W/m/K')
    
    print("Done!")

if __name__ == '__main__':
    validate_air()
