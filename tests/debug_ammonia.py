
import json
import numpy as np
import jax.numpy as jnp
from chillprop import parameters, transport
import CoolProp.CoolProp as CP

def debug_ammonia():
    path = r"c:\Users\Benji\Documents\ChillProp\CoolProp\dev\fluids\Ammonia.json"
    with open(path, 'r') as f:
        data = json.load(f)
    params = parameters.FluidParameters.from_json(data)
    
    # Test point
    Tc = params.Tc
    rhoc = params.rhoc
    T = 1.1 * Tc
    rho = 0.5 * rhoc
    
    print(f"Ammonia Test Point:")
    print(f"T = {T} K")
    print(f"rho = {rho} mol/m^3")
    
    # Viscosity Components
    eta0 = transport.viscosity_dilute(params, jnp.array(T))
    print(f"eta0 (Dilute): {eta0} Pa-s")
    
    v = params.viscosity
    eta_initial = 0.0
    if v.initial_density:
        rf = v.initial_density
        Tstar = T / v.epsilon_over_k
        B_eta = jnp.sum(rf.b * (Tstar ** rf.t))
        print(f"Rainwater-Friend B_eta: {B_eta}")
        eta_initial = eta0 * B_eta * rho
        print(f"eta_initial: {eta_initial} Pa-s")
        
    eta_ho = 0.0
    if v.higher_order:
        ho = v.higher_order
        rho_r = ho['rhomolar_reduce']
        T_r = ho['T_reduce']
        delta = rho / rho_r
        tau = T_r / T
        
        S_ho = jnp.sum(ho['a'] * (delta ** ho['d1']) * (tau ** ho['t1']) * jnp.exp(ho['gamma'] * (delta ** ho['l'])))
        F = jnp.sum(ho['f'] * (delta ** ho['d2']) * (tau ** ho['t2']))
        delta0 = jnp.sum(ho['g'] * (tau ** ho['h'])) / jnp.sum(ho['p'] * (tau ** ho['q']))
        
        eta_ho = S_ho + F * (1.0 / (delta0 - delta) - 1.0 / delta0)
        print(f"eta_higher_order: {eta_ho} Pa-s")
        
    eta_total = eta0 + eta_initial + eta_ho
    print(f"eta_total: {eta_total} Pa-s")
    
    eta_cp = CP.PropsSI("viscosity", "T", T, "D", rho, "Ammonia")
    print(f"CoolProp eta: {eta_cp} Pa-s")
    print(f"Error: {abs(eta_total - eta_cp)/eta_cp:.1%}")

if __name__ == "__main__":
    debug_ammonia()
