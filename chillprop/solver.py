import jax
import jax.numpy as jnp
import equinox as eqx
from chillprop.parameters import FluidParameters
from chillprop.core import pressure

def find_rho_PT(params: FluidParameters, P_target: jax.Array, T: jax.Array, rho_guess: jax.Array, max_iter: int = 20, tol: float = 1e-9) -> jax.Array:
    """
    Find density rho for a given Pressure and Temperature using Newton-Raphson.
    f(rho) = P(rho, T) - P_target = 0
    """
    
    def step(rho, _):
        P = pressure(params, rho, T)
        # dP/drho at constant T
        dP_drho = jax.grad(pressure, argnums=1)(params, rho, T)
        
        delta_rho = (P - P_target) / dP_drho
        new_rho = rho - delta_rho
        
        # Check convergence? In JAX we usually run fixed iterations or use jax.lax.while_loop.
        # For a drop-in replacement, robustness is key.
        error = jnp.abs(new_rho - rho) / rho
        return new_rho, error

    # Use a fixed-iteration scan for simplicity and JIT-friendliness, 
    # but we could use while_loop for efficiency if needed.
    # Note: To enable implicit differentiation later, we might need a custom root finder.
    
    rho_final, errors = jax.lax.scan(step, rho_guess, jnp.arange(max_iter))
    
    return rho_final

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
    # If P < Psat, we are superheated vapor -> Use rho_ideal
    # (rho_V_anc is only good at Psat; rho_ideal is good globally for gas)
    is_liquid_pressure = P > Psat
    guess_sub = jnp.where(is_liquid_pressure, rho_L_anc, rho_ideal)
    
    # Supercritical Logic:
    # If P > Pc, we are dense supercritical fluid -> Guess rho_critical as a safe anchor?
    # actually ideal gas is very bad for high P.
    # Let's use rho_c for high pressure, ideal gas for low pressure.
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
        # we can't use jax.jacobian(func) directly if func is not jittable or has non-array args?
        # params is common.
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

