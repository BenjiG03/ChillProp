import jax
import jax.numpy as jnp
import equinox as eqx
from chillprop.parameters import FluidParameters, AncillaryEquation
from chillprop.core import pressure, alpha0, alphar

def evaluate_ancillary(anc: AncillaryEquation, T: jax.Array) -> jax.Array:
    """Evaluate CoolProp ancillary equation."""
    theta = 1.0 - T / anc.T_r
    sigma = jnp.sum(anc.n * (theta ** anc.t))
    
    val = jnp.where(
        (anc.type == 'pL') | (anc.type == 'rhoV'),
        anc.reducing_value * jnp.exp((anc.T_r / T) * sigma),
        jnp.where(
            anc.type == 'rhoLnoexp',
            anc.reducing_value * (1.0 + sigma),
            0.0
        )
    )
    return val

def rhol_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    if params.ancillary_rhoL is None: return jnp.nan
    return evaluate_ancillary(params.ancillary_rhoL, T)

def rhov_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    if params.ancillary_rhoV is None: return jnp.nan
    return evaluate_ancillary(params.ancillary_rhoV, T)

def psat_anc(params: FluidParameters, T: jax.Array) -> jax.Array:
    if params.ancillary_p is None: return jnp.nan
    return evaluate_ancillary(params.ancillary_p, T)

def chemical_potential(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Molar chemical potential mu = A + PV = RT(alpha + 1 + delta*alphar_delta)"""
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
    T = jnp.asarray(T)
    rl_guess = rhol_anc(params, T)
    rv_guess = rhov_anc(params, T)
    
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
        new_x = x - delta_x
        
        return new_x, None

    x_final, _ = jax.lax.scan(step, jnp.array([rl_guess, rv_guess]), jnp.arange(max_iter))
    return x_final

def vapor_quality(rho: jax.Array, rho_l: jax.Array, rho_v: jax.Array) -> jax.Array:
    """Calculate vapor quality q."""
    v = 1.0 / rho
    vl = 1.0 / rho_l
    vv = 1.0 / rho_v
    return (v - vl) / (vv - vl)

@eqx.filter_jit
def get_phase(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Determine phase."""
    Tc = params.Tc
    is_supercritical = T > Tc
    
    rho_l, rho_v = solve_vle(params, T)
    
    is_twophase = (T <= Tc) & (rho > rho_v) & (rho < rho_l)
    is_liquid = (T <= Tc) & (rho >= rho_l)
    
    phase = jnp.where(is_supercritical, 3, 
                jnp.where(is_twophase, 2,
                    jnp.where(is_liquid, 0, 1)))
    return phase
