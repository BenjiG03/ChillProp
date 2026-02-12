import jax
import jax.numpy as jnp
from chillprop.parameters import FluidParameters

def viscosity_dilute(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Dilute gas viscosity [Pa-s]"""
    if params.viscosity is None: return jnp.nan
    v = params.viscosity
    d = v.dilute
    
    Tstar = T / v.epsilon_over_k
    lnTstar = jnp.log(Tstar)
    
    S_log = jnp.sum(d['a'] * (lnTstar ** d['t']))
    S = jnp.exp(S_log)
    
    sigma_nm = v.sigma_eta * 1e9
    molar_mass_kgkmol = params.M * 1000.0 # kg/mol to g/mol (kg/kmol)
    
    C = d.get('C', 26.692e-9)
    eta0 = C * jnp.sqrt(molar_mass_kgkmol * T) / (sigma_nm**2 * S)
    return eta0

def viscosity(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Dynamic Viscosity [Pa-s]"""
    if params.viscosity is None: return jnp.nan
    
    eta0 = viscosity_dilute(params, T)
    
    # Higher order part (modified_Batschinski_Hildebrand)
    v = params.viscosity
    if v.higher_order is None: return eta0
    
    ho = v.higher_order
    rho_r = ho['rhomolar_reduce']
    T_r = ho['T_reduce']
    delta = rho / rho_r
    tau = T_r / T
    
    # S = sum(a * delta^d1 * tau^t1 * exp(gamma * delta^l))
    S_ho = jnp.sum(ho['a'] * (delta ** ho['d1']) * (tau ** ho['t1']) * jnp.exp(ho['gamma'] * (delta ** ho['l'])))
    
    # F = sum(f * delta^d2 * tau^t2)
    F = jnp.sum(ho['f'] * (delta ** ho['d2']) * (tau ** ho['t2']))
    
    # delta0 = sum(g * tau^h) / sum(p * tau^q)
    delta0 = jnp.sum(ho['g'] * (tau ** ho['h'])) / jnp.sum(ho['p'] * (tau ** ho['q']))
    
    eta_ho = S_ho + F * (1.0 / (delta0 - delta) - 1.0 / delta0)
    
    return eta0 + eta_ho

def thermal_conductivity(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Thermal Conductivity [W/m/K]"""
    if params.conductivity is None: return jnp.nan
    
    c = params.conductivity
    tau = params.Tc / T
    delta = rho / params.rhoc
    
    # 1. Dilute (eta0_and_poly)
    eta0_uPas = viscosity_dilute(params, T) * 1e6
    d = c.dilute
    lambda_dilute = d['A'][0] * eta0_uPas
    if d['A'].shape[0] > 1:
        lambda_dilute += jnp.sum(d['A'][1:] * (tau ** d['t'][1:]))
    
    # 2. Residual (polynomial_and_exponential)
    res = c.residual
    lambda_res = jnp.sum(res['A'] * (tau ** res['t']) * (delta ** res['d']) * jnp.exp(-res['gamma'] * (delta ** res['l'])))
    
    # 3. Critical (simplified_Olchowy_Sengers) - skipped for now
    
    return lambda_dilute + lambda_res
