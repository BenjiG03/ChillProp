import jax
import jax.numpy as jnp
import equinox as eqx
from chillprop.parameters import FluidParameters, AncillaryEquation
from chillprop.core import pressure, alpha0, alphar

def evaluate_ancillary(anc: AncillaryEquation, T: jax.Array) -> jax.Array:
    """Evaluate CoolProp ancillary equation."""
    theta = 1.0 - T / anc.T_r
    sigma = jnp.sum(anc.n * (theta ** anc.t))
    
    # If using_tau_r is true, scale exponent by Tr/T
    exponent = jnp.where(anc.using_tau_r, sigma * (anc.T_r / T), sigma)
    
    val = jnp.where(
        (anc.type == 'pL') | (anc.type == 'pV') | (anc.type == 'rhoV') | (anc.type == 'pS'),
        anc.reducing_value * jnp.exp(exponent),
        jnp.where(
            anc.type == 'rhoLnoexp',
            anc.reducing_value * (1.0 + sigma),
            0.0
        )
    )
    return val

def rhol_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Return ancillary saturated-liquid density."""
    if params.ancillary_rhoL is None: return jnp.nan
    return evaluate_ancillary(params.ancillary_rhoL, T)

def rhov_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Return ancillary saturated-vapor density."""
    if params.ancillary_rhoV is None: return jnp.nan
    return evaluate_ancillary(params.ancillary_rhoV, T)

def psat_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Return ancillary saturation pressure."""
    if params.ancillary_pS is not None:
        return evaluate_ancillary(params.ancillary_pS, T)
    if params.ancillary_pL is not None:
        # Default to bubble point if pure Psatt is missing
        return evaluate_ancillary(params.ancillary_pL, T)
    if params.ancillary_pV is not None:
        return evaluate_ancillary(params.ancillary_pV, T)
    return jnp.nan

def chemical_potential(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Return molar chemical potential."""
    tau = params.Tc / T
    delta = rho / params.rhoc
    
    # helper for gradients
    dar_ddelta = jax.grad(alphar, argnums=2)(params, tau, delta)
    a0_val = alpha0(params, tau, delta)
    ar_val = alphar(params, tau, delta)
    
    return params.R * T * (a0_val + ar_val + 1.0 + delta * dar_ddelta)

@eqx.filter_jit
def solve_vle(params: FluidParameters, T: jax.Array, max_iter: int = 20) -> jax.Array:
    """
    Solve for (rho_liq, rho_vap) at temperature T using Newton-Raphson.
    """
    rl_guess = rhol_anc(params, T)
    rv_guess = rhov_anc(params, T)
    
    if params.pseudo_pure:
        return jnp.array([rl_guess, rv_guess])

    ps_guess = psat_anc(params, T)
    from chillprop.solver import find_rho_PT

    rl_seed = find_rho_PT(params, ps_guess, T, rl_guess)
    rv_seed = find_rho_PT(params, ps_guess, T, rv_guess)
    
    def step(x, _):
        rl, rv = x[0], x[1]
        
        pl = pressure(params, rl, T)
        pv = pressure(params, rv, T)
        mul = chemical_potential(params, rl, T)
        muv = chemical_potential(params, rv, T)
        
        f_val = jnp.array([pl - pv, mul - muv])
        
        dpl_drl = jax.grad(pressure, argnums=1)(params, rl, T)
        dpv_drv = jax.grad(pressure, argnums=1)(params, rv, T)
        dmul_drl = jax.grad(chemical_potential, argnums=1)(params, rl, T)
        dmuv_drv = jax.grad(chemical_potential, argnums=1)(params, rv, T)
        
        jac = jnp.array([
            [dpl_drl, -dpv_drv],
            [dmul_drl, -dmuv_drv]
        ])
        
        delta_x = jnp.linalg.solve(jac, f_val)
        new_x = jnp.maximum(x - delta_x, 1e-12)
        
        return new_x, None

    x_final, _ = jax.lax.scan(step, jnp.array([rl_seed, rv_seed]), jnp.arange(max_iter))
    return x_final

def vapor_quality(rho: jax.Array, rho_l: jax.Array, rho_v: jax.Array) -> jax.Array:
    """Calculate vapor quality from bulk and saturation densities."""
    v = 1.0 / rho
    vl = 1.0 / rho_l
    vv = 1.0 / rho_v
    return (v - vl) / (vv - vl)

@eqx.filter_jit
def get_phase(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Classify the phase as liquid, vapor, two-phase, or supercritical."""
    Tc = params.Tc
    is_supercritical = T > Tc
    
    rho_l, rho_v = solve_vle(params, T)
    
    is_twophase = (T <= Tc) & (rho > rho_v) & (rho < rho_l)
    is_liquid = (T <= Tc) & (rho >= rho_l)
    
    phase = jnp.where(is_supercritical, 3, 
                jnp.where(is_twophase, 2,
                    jnp.where(is_liquid, 0, 1)))
    return phase
