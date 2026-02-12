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

@eqx.filter_jit
def solve_rho_PT(params: FluidParameters, P: jax.Array, T: jax.Array) -> jax.Array:
    """
    High-level density solver with guess generation.
    """
    # Simple ideal gas guess: rho = P / (params.R * T)
    rho_guess = P / (params.R * T)
    
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

