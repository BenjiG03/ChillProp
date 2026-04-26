import jax
import jax.numpy as jnp
import equinox as eqx
from chillprop.parameters import FluidParameters
from chillprop.core import pressure

def find_rho_PT(params: FluidParameters, P_target: jax.Array, T: jax.Array, rho_guess: jax.Array, max_iter: int = 50, tol: float = 1e-9) -> jax.Array:
    """
    Find density rho for a given Pressure and Temperature using Newton-Raphson.
    f(rho) = P(rho, T) - P_target = 0
    """
    import equinox as eqx
    dyn_params, static_params = eqx.partition(params, eqx.is_array)
    
    @jax.custom_vjp
    def inner_find_rho(dyn_p, P_t, T_val, r_guess):
        p = eqx.combine(dyn_p, static_params)
        def step(rho, _):
            P = pressure(p, rho, T_val)
            dP_drho = jax.grad(pressure, argnums=1)(p, rho, T_val)
            delta_rho = (P - P_t) / dP_drho
            return rho - delta_rho, None
            
        rho_final, _ = jax.lax.scan(step, r_guess, jnp.arange(max_iter))
        return rho_final

    def fwd(dyn_p, P_t, T_val, r_guess):
        rho_final = inner_find_rho(dyn_p, P_t, T_val, r_guess)
        return rho_final, (dyn_p, rho_final, P_t, T_val)

    def bwd(res, g):
        dyn_p, rho_final, P_t, T_val = res
        p = eqx.combine(dyn_p, static_params)
        
        dP_drho = jax.grad(pressure, argnums=1)(p, rho_final, T_val)
        dP_dT = jax.grad(pressure, argnums=2)(p, rho_final, T_val)
        
        g_P_target = g / dP_drho
        g_T = g * (-dP_dT / dP_drho)
        g_rho_guess = jnp.zeros_like(rho_final)
        g_dyn = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, dyn_p)
        
        return (g_dyn, g_P_target, g_T, g_rho_guess)

    inner_find_rho.defvjp(fwd, bwd)
    
    return inner_find_rho(dyn_params, P_target, T, rho_guess)


from chillprop.phases import rhol_anc, rhov_anc, psat_anc

@eqx.filter_jit
def solve_rho_PT(params: FluidParameters, P: jax.Array, T: jax.Array) -> jax.Array:
    """
    High-level density solver with phase-aware guess generation.
    """
    # 1. Calculate Ancillary (Saturation) Properties
    # Clip T to be slightly below Tc for ancillary evaluation to avoid NaNs in supercritical region
    # (JAX evaluates both branches of where)
    T_sat = jnp.minimum(T, params.Tc - 0.001)
    
    Psat = psat_anc(params, T_sat)
    rho_L_anc = rhol_anc(params, T_sat)
    rho_V_anc = rhov_anc(params, T_sat)
    rho_ideal = P / (params.R * T)
    
    # 2. Determine Phase Regime for Guesses
    is_subcritical = T < params.Tc
    
    # Subcritical Logic: Compare P to Psat
    # If P > Psat, we are compressed liquid -> Use rho_L_anc
    # Superheated vapor -> Use scaled rho_V_anc for better accuracy near saturation
    # rho approx P/Psat * rho_V_anc (assuming Z constant along isotherm)
    # Safeguard against Psat being 0 at very low temperatures
    Psat_safe = jnp.maximum(Psat, 1e-10)
    rho_vapor_guess = jnp.minimum(rho_V_anc * (P / Psat_safe), params.rhoc)
    
    is_liquid_pressure = P > Psat
    guess_sub = jnp.where(is_liquid_pressure, rho_L_anc, rho_vapor_guess)
    
    # Supercritical logic: use rho_c for high pressures where the ideal-gas guess breaks down, otherwise ideal gas.
    is_high_pressure = P > params.Pc
    guess_sup = jnp.where(is_high_pressure, params.rhoc, rho_ideal)

    # Combine guesses
    rho_guess = jnp.where(is_subcritical, guess_sub, guess_sup)
    
    # Fallback: if ancillary returns NaN (e.g. valid range issues), use ideal gas
    rho_guess = jnp.where(jnp.isnan(rho_guess), rho_ideal, rho_guess)
    
    # Run Newton solver
    return find_rho_PT(params, P, T, rho_guess)

def solve_2d(params: FluidParameters, func, target, guess, max_iter: int = 20):
    """
    Generic 2D Newton solver for F(X) = target.
    func(rho, T) -> list/array of 2 values.
    """
    def step(x, _):
        rho, T = x[0], x[1]
        
        # F(X) - target
        f_val = jnp.array(func(params, rho, T)) - jnp.array(target)
        
        # Jacobian dF/dX
        # Wrap func so jax.jacobian always receives array inputs even though params are shared.
        def f_wrapper(x_vec):
            return jnp.array(func(params, x_vec[0], x_vec[1]))
            
        jac = jax.jacobian(f_wrapper)(x)
        
        delta_x = jnp.linalg.solve(jac, f_val)
        new_x = x - delta_x
        
        return new_x, None

    x_final, _ = jax.lax.scan(step, jnp.array(guess), jnp.arange(max_iter))
    return x_final

from chillprop.core import enthalpy, entropy

@eqx.filter_jit
def solve_rhoT_Ph(params: FluidParameters, P: jax.Array, h: jax.Array) -> jax.Array:
    """Solve for (rho, T) given (P, h)"""
    # Guess: T approx 300, rho approx P/RT
    T_guess = 300.0
    rho_guess = P / (params.R * T_guess)
    
    def ph_func(p, r, t):
        return [pressure(p, r, t), enthalpy(p, r, t)]
        
    return solve_2d(params, ph_func, [P, h], [rho_guess, T_guess])

@eqx.filter_jit
def solve_rhoT_Ps(params: FluidParameters, P: jax.Array, s: jax.Array) -> jax.Array:
    """Solve for (rho, T) given (P, s)"""
    T_guess = 350.0
    rho_guess = P / (params.R * T_guess)
    
    def ps_func(p, r, t):
        return [pressure(p, r, t), entropy(p, r, t)]
        
    return solve_2d(params, ps_func, [P, s], [rho_guess, T_guess])

