import jax
import jax.numpy as jnp
from chillprop.parameters import (
    FluidParameters,
    IdealHelmholtzLead,
    IdealHelmholtzLogTau,
    IdealHelmholtzPower,
    IdealHelmholtzPlanckEinstein,
    IdealHelmholtzPlanckEinsteinFunctionT,
    ResidualHelmholtzPower,
    ResidualHelmholtzGaussian
)

def alpha0_lead(term: IdealHelmholtzLead, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # a1 + a2*tau
    # Note: CoolProp LogTau uses log(tau). Lead uses linear?
    # Checking CoolProp source or docs:
    # Lead term: a1 + a2*tau (usually related to enthalpy/entropy integration constants)
    # Actually, often it is log(delta) + a1 + a2*tau.
    # The log(delta) comes from ideal gas law integration?
    # CoolProp separates the log(delta) term usually?
    # Let's check CoolProp docs. 
    # Ideal helmholtz energy alpha0 = h0/(RT) - s0/R - 1 + ln(delta/tau_0?) ...
    # Usually a term 'log(delta)' is implicit for ideal gas?
    # Wait, the JSON had 'IdealGasHelmholtzLead'.
    # I should check logic.
    # But strictly evaluating the TERM:
    # It is likely a1 + a2*tau.
    return term.a1 + term.a2 * tau

def alpha0_logtau(term: IdealHelmholtzLogTau, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # a * log(tau)
    return term.a * jnp.log(tau)

def alpha0_power(term: IdealHelmholtzPower, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # sum(n_i * tau^t_i)
    return jnp.sum(term.n * (tau ** term.t))

def alpha0_planck_einstein(term: IdealHelmholtzPlanckEinstein, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # sum(n_i * log(1 - exp(-gamma_i * tau))) where gamma_i is theta_i / Tc?
    # term.t is "t". In CoolProp JSON it was "t".
    # Usually t = theta/Tc.
    # So exp(-t * tau) ? No, tau = Tc/T.
    # theta/T = theta/Tc * Tc/T = t * tau.
    # So log(1 - exp(-t * tau))
    vals = term.n * jnp.log(1 - jnp.exp(-term.t * tau))
    return jnp.sum(vals)

def alpha0_planck_einstein_function_t(term: IdealHelmholtzPlanckEinsteinFunctionT, tau: jax.Array, delta: jax.Array, Tc: float) -> jax.Array:
    # term.v is "v" (theta).
    # theta/T = v / T = v / (Tc / tau) = v * tau / Tc
    # Term is n * log(1 - exp(-theta/T))
    # It seems logic is same, just parameterization differs (absolute v vs dimensionless t).
    theta = term.v
    val = term.n * jnp.log(1 - jnp.exp(-theta * tau / Tc))
    return jnp.sum(val)

def alphar_power(term: ResidualHelmholtzPower, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # n * delta^d * tau^t * exp(-delta^l)
    # If l=0, it is a polynomial term: n * delta^d * tau^t (exp factor is 1)
    # If l>0, it is an exponential term.
    # Note: delta^0 = 1. exp(-1) != 1. So we must conditionally apply exp.
    
    # Calculate exponential part
    # We use jnp.where to handle l=0 vs l!=0
    # exp_part = exp(-delta^l) if l!=0 else 1.0
    
    # To handle gradients correctly, we should execute the power only if needed?
    # Or just use math: if l=0, we want 1. 
    # Can we shift l? No.
    # jnp.where is safe.
    
    exp_factor = jnp.where(term.l != 0, jnp.exp(-(delta ** term.l)), 1.0)
    
    val = term.n * (delta ** term.d) * (tau ** term.t) * exp_factor
    return jnp.sum(val)

def alphar_gaussian(term: ResidualHelmholtzGaussian, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # n * delta^d * tau^t * exp(-eta*(delta-epsilon)^2 - beta*(tau-gamma)^2)
    val = term.n * (delta ** term.d) * (tau ** term.t) * jnp.exp(
        -term.eta * ((delta - term.epsilon)**2) - term.beta * ((tau - term.gamma)**2)
    )
    return jnp.sum(val)

def alpha0(params: FluidParameters, tau: jax.Array, delta: jax.Array) -> jax.Array:
    tau = jnp.asarray(tau)
    delta = jnp.asarray(delta)
    val = jnp.log(delta) # The ideal gas logarithmic density term (required for P = rho RT). 
    # Wait, does CoolProp include log(delta) in alpha0 sum or separate?
    # Usually it is separate.
    # alpha = alpha0 + alphar + ln(delta)
    # But sometimes alpha0 includes ln(delta).
    # Let's check "IdealGasHelmholtzLead" documentation.
    # If I verify parity, I must match CoolProp.
    # CoolProp's alpha0 function typically includes the summation of terms. NOT ln(delta)?
    # BUT, P = rho RT (1 + delta * alphar_delta). 
    # alpha0 is not used for P calculation!
    # alpha0 is used for h, s.
    # s = R(tau alpha0_tau - alpha0 ...)
    # If alpha0 has ln(delta), then alpha0_tau doesn't change?
    # But s depends on ln(delta) ?
    # s_ideal = s0 - R ln(rho/rho0) ...
    # This implies ln(delta) term is needed.
    
    # I will assume `val = jnp.log(delta)` is part of the TOTAL alpha0, but maybe not the stored TERMS.
    # I'll implement `evaluate_alpha0` which sums terms + log(delta).
    
    for term in params.alpha0:
        if isinstance(term, IdealHelmholtzLead):
            val += alpha0_lead(term, tau, delta)
        elif isinstance(term, IdealHelmholtzLogTau):
            val += alpha0_logtau(term, tau, delta)
        elif isinstance(term, IdealHelmholtzPower):
            val += alpha0_power(term, tau, delta)
        elif isinstance(term, IdealHelmholtzPlanckEinstein):
            val += alpha0_planck_einstein(term, tau, delta)
        elif isinstance(term, IdealHelmholtzPlanckEinsteinFunctionT):
            val += alpha0_planck_einstein_function_t(term, tau, delta, params.Tc)
            
    return val

def alphar(params: FluidParameters, tau: jax.Array, delta: jax.Array) -> jax.Array:
    tau = jnp.asarray(tau)
    delta = jnp.asarray(delta)
    val = jnp.array(0.0, dtype=tau.dtype)
    for term in params.alphar:
        if isinstance(term, ResidualHelmholtzPower):
            val += alphar_power(term, tau, delta)
        elif isinstance(term, ResidualHelmholtzGaussian):
            val += alphar_gaussian(term, tau, delta)
    return val
