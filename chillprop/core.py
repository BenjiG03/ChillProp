import jax
import jax.numpy as jnp
from chillprop.parameters import FluidParameters
from chillprop.heos import alpha0, alphar

def get_alpha_and_derivs(params: FluidParameters, rho: jax.Array, T: jax.Array):
    tau = params.Tr / T
    delta = rho / params.rhor
    
    # We need partials w.r.t tau and delta
    # alpha0(params, tau, delta)
    # alphar(params, tau, delta)
    
    # helper for gradients
    def a0_func(t, d): return alpha0(params, t, d)
    def ar_func(t, d): return alphar(params, t, d)
    
    # Value and grad
    val_a0, (da0_dtau, da0_ddelta) = jax.value_and_grad(a0_func, argnums=(0, 1))(tau, delta)
    val_ar, (dar_dtau, dar_ddelta) = jax.value_and_grad(ar_func, argnums=(0, 1))(tau, delta)
    
    return {
        'tau': tau,
        'delta': delta,
        'a0': val_a0,
        'ar': val_ar,
        'da0_dtau': da0_dtau,
        'da0_ddelta': da0_ddelta,
        'dar_dtau': dar_dtau,
        'dar_ddelta': dar_ddelta
    }

def pressure(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    # P = rho * R * T * (1 + delta * alphar_delta)
    # Note: alpha0 does not affect Pressure directly (only ideal gas part which is P=rhoRT)
    # The term delta * alphar_delta accounts for non-ideality.
    # What about alpha0 derivative w.r.t delta? 
    # alpha0 = log(delta) + ...
    # da0_ddelta = 1/delta
    # If we used total alpha aka alpha = a0 + ar.
    # P = rho^2 * R * T * (d_alpha_drho) ?
    # P = - (dF/dV)_T = ... = rho^2 * (dA/drho)_T
    # A = R T alpha.
    # dA/drho = R T * (dalpha/ddelta * ddelta/drho) = RT * alpha_delta * (1/rhoc).
    # P = rho^2 * RT * alpha_delta / rhoc = rho * (rho/rhoc) * RT * alpha_delta = rho * delta * RT * alpha_delta.
    # alpha_delta = da0_ddelta + dar_ddelta = 1/delta + dar_ddelta.
    # P = rho * RT * delta * (1/delta + dar_ddelta) = rho * RT * (1 + delta * dar_ddelta).
    # This matches.
    
    # We can calculate just ar derivatives for efficiency if we only need P.
    tau = params.Tr / T
    delta = rho / params.rhor
    
    dar_ddelta = jax.grad(alphar, argnums=2)(params, tau, delta)
    
    return rho * params.R * T * (1.0 + delta * dar_ddelta)

def enthalpy(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    # h = R T (1 + tau(da0_dtau + dar_dtau) + delta * dar_ddelta)
    vals = get_alpha_and_derivs(params, rho, T)
    tau = vals['tau']
    delta = vals['delta']
    return params.R * T * (1.0 + tau * (vals['da0_dtau'] + vals['dar_dtau']) + delta * vals['dar_ddelta'])

def entropy(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    # s = R (tau(da0_dtau + dar_dtau) - (a0 + ar))
    vals = get_alpha_and_derivs(params, rho, T)
    tau = vals['tau']
    return params.R * (tau * (vals['da0_dtau'] + vals['dar_dtau']) - (vals['a0'] + vals['ar']))

def internal_energy(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    # u = R T * tau * (da0_dtau + dar_dtau)
    vals = get_alpha_and_derivs(params, rho, T)
    tau = vals['tau']
    return params.R * T * tau * (vals['da0_dtau'] + vals['dar_dtau'])

def cvmolar(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    tau = params.Tr / T
    delta = rho / params.rhor
    
    # helper for grads
    def alpha_total(t, d):
        return alpha0(params, t, d) + alphar(params, t, d)
    
    # d2alpha_dtau2
    d2 = jax.grad(jax.grad(alpha_total, argnums=0), argnums=0)(tau, delta)
    return -params.R * (tau**2) * d2

def cpmolar(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    tau = params.Tr / T
    delta = rho / params.rhor
    
    # Need several derivatives
    def a_tot(t, d): return alpha0(params, t, d) + alphar(params, t, d)
    def ar(t, d): return alphar(params, t, d)
    
    # C_v part
    cv = cvmolar(params, rho, T)
    
    # Derivatives for Cp-Cv
    dar_ddelta = jax.grad(ar, argnums=1)(tau, delta)
    dar_ddelta2 = jax.grad(jax.grad(ar, argnums=1), argnums=1)(tau, delta)
    dar_ddelta_dtau = jax.grad(jax.grad(ar, argnums=1), argnums=0)(tau, delta)
    
    num = (1.0 + delta * dar_ddelta - delta * tau * dar_ddelta_dtau)**2
    den = (1.0 + 2.0 * delta * dar_ddelta + delta**2 * dar_ddelta2)
    
    return cv + params.R * num / den

def props(params: FluidParameters, rho: jax.Array, T: jax.Array) -> dict:
    vals = get_alpha_and_derivs(params, rho, T)
    tau = vals['tau']
    delta = vals['delta']
    
    R = params.R
    
    # Common terms
    tau_alpha_tau = tau * (vals['da0_dtau'] + vals['dar_dtau'])
    delta_alpha_delta = delta * vals['dar_ddelta']
    
    p = rho * R * T * (1.0 + delta_alpha_delta)
    u = R * T * tau_alpha_tau
    h = R * T * (1.0 + tau_alpha_tau + delta_alpha_delta)
    s = R * (tau_alpha_tau - (vals['a0'] + vals['ar']))
    
    return {'p': p, 'u': u, 'h': h, 's': s}
