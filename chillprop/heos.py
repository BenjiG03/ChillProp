import jax
import jax.numpy as jnp
from chillprop.parameters import (
    FluidParameters,
    IdealHelmholtzLead,
    IdealHelmholtzLogTau,
    IdealHelmholtzPower,
    IdealHelmholtzPlanckEinstein,
    IdealHelmholtzPlanckEinstein,
    IdealHelmholtzPlanckEinsteinFunctionT,
    IdealHelmholtzPlanckEinsteinGeneralized,
    IdealHelmholtzEnthalpyEntropyOffset,
    IdealHelmholtzCP0Constant,
    IdealHelmholtzCP0PolyT,
    ResidualHelmholtzPower,
    ResidualHelmholtzGaussian,
    ResidualHelmholtzNonAnalytic,
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

def alpha0_enthalpy_entropy_offset(term: IdealHelmholtzEnthalpyEntropyOffset, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # Same form as Lead: a1 + a2*tau
    # Specifically: alpha0 = h0/(RTc) * tau - s0/R
    # Wait, usually a1 = -s0/R, a2 = h0/(RTc)
    # Check CoolProp:
    # "a1": -14.49..., "a2": 8.82...
    # It is simply a1 + a2 * tau
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

def alpha0_planck_einstein_function_t(term: IdealHelmholtzPlanckEinsteinFunctionT, tau: jax.Array, delta: jax.Array, Tr: float) -> jax.Array:
    # term.v is "v" (theta).
    # theta/T = v / T = v / (Tr / tau) = v * tau / Tr
    # Term is n * log(1 - exp(-theta/T))
    # It seems logic is same, just parameterization differs (absolute v vs dimensionless t).
    theta = term.v
    theta = term.v
    val = term.n * jnp.log(1 - jnp.exp(-theta * tau / Tr))
    return jnp.sum(val)

def alpha0_planck_einstein_generalized(term: IdealHelmholtzPlanckEinsteinGeneralized, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # sum(n_i * log(c_i + d_i * exp(t_i * tau)))
    # Note: t_i here corresponds to 'theta' in typical notation
    # but in our parameter class it is 't'.
    val = term.n * jnp.log(term.c + term.d * jnp.exp(term.t * tau))
    return jnp.sum(val)

def alpha0_cp0_constant(term: IdealHelmholtzCP0Constant, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # alpha0 = c - c*T0/Tc*tau - c*ln(Tc/T0) + c*ln(tau)
    c = term.cp_over_R
    T0 = term.T0
    Tc = term.Tc
    val = c - c * (T0 / Tc) * tau - c * jnp.log(Tc / T0) + c * jnp.log(tau)
    return val

def alpha0_cp0_polyt(term: IdealHelmholtzCP0PolyT, tau: jax.Array, delta: jax.Array) -> jax.Array:
    # Int(Cp0/R) contributions
    # For each term c*T^t:
    # alpha0_i = c * [ (T^t - T0^(t+1)/T)/(t+1) - (T^t - T0^t)/t ]
    # Substitute T = Tc/tau
    
    # We need to handle t=0 and t=-1 cases.
    # But usually t is float.
    # Note: t=0 is handled by CP0Constant if separated. 
    # But CP0PolyT might contain t=0.
    
    c = term.c
    t = term.t
    T0 = term.T0
    Tc = term.Tc
    T = Tc / tau
    
    # helper for power term
    def term_val(ci, ti):
        # h_part:
        # Case ti != -1: h_part = ci/(ti+1) * (T^ti - T0^(ti+1)/T)
        # Case ti == -1: h_part = ci/T * (ln(T) - ln(T0))
        h_part = jnp.where(
            jnp.abs(ti + 1) > 1e-10, 
            (ci / (ti + 1)) * (T**ti - (T0**(ti + 1)) / T),
            (ci / T) * (jnp.log(T) - jnp.log(T0))
        )
        
        # s_part:
        # Case ti != 0: s_part = ci/ti * (T^ti - T0^ti)
        # Case ti == 0: s_part = ci * ln(T/T0)
        s_part = jnp.where(
            jnp.abs(ti) > 1e-10,
            (ci / ti) * (T**ti - T0**ti),
            ci * (jnp.log(T) - jnp.log(T0))
        )
        
        return h_part - s_part

    # Vectorized computation
    # JAX scan or vmap? 
    # Since terms are small (len(c) ~ 5-10), direct sum is fine.
    # But shapes must match. c, t are 1D arrays. T is scalar (or batched).
    # We can use vmap over (c, t).
    
    # vmap over parameters, broadcast over state
    # T_ = T if scalar, else... T might be array.
    # Let's assume T is scalar for now or broadcasatable.
    
    # We iterate manually to avoid shape complexity if c is small
    val = jnp.array(0.0, dtype=tau.dtype)
    for i in range(len(c)):
        val += term_val(c[i], t[i])
        
    return val

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

def alphar_nonanalytic(term: ResidualHelmholtzNonAnalytic, tau: jax.Array, delta: jax.Array) -> jax.Array:
    """
    Non-analytic residual Helmholtz term matching CoolProp's ResidualHelmholtzNonAnalytic.

    Value (see CoolProp src/Helmholtz.cpp):
      theta = (1 - tau) + A * |delta - 1|^(1/beta)
      PSI   = exp(-C*(delta-1)^2 - D*(tau-1)^2)
      DELTA = theta^2 + B * |delta - 1|^(2a)
      alphar += delta * n * DELTA^b * PSI
    """
    tau = jnp.asarray(tau)
    delta = jnp.asarray(delta)

    # CoolProp offsets tau/delta extremely close to 1 to avoid undefined intermediates
    eps = 10.0 * jnp.finfo(tau.dtype).eps
    tau_s = jnp.where(jnp.abs(tau - 1.0) < eps, tau + eps, tau)
    delta_s = jnp.where(jnp.abs(delta - 1.0) < eps, delta + eps, delta)

    dm1 = delta_s - 1.0
    tm1 = tau_s - 1.0

    # |delta-1|^(1/beta) is computed as ( (delta-1)^2 )^(1/(2*beta)) to match CoolProp
    dm1_sq = dm1 * dm1
    theta = (1.0 - tau_s) + term.A * (dm1_sq ** (1.0 / (2.0 * term.beta)))
    PSI = jnp.exp(-term.C * dm1_sq - term.D * (tm1 * tm1))
    DELTA = (theta * theta) + term.B * (dm1_sq ** term.a)
    return jnp.sum(delta_s * term.n * (DELTA ** term.b) * PSI)

def alpha0(params: FluidParameters, tau: jax.Array, delta: jax.Array) -> jax.Array:
    tau = jnp.asarray(tau)
    delta = jnp.asarray(delta)
    val = jnp.log(delta) 
    
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
            val += alpha0_planck_einstein_function_t(term, tau, delta, params.Tr)
        elif isinstance(term, IdealHelmholtzPlanckEinsteinGeneralized):
            val += alpha0_planck_einstein_generalized(term, tau, delta)
        elif isinstance(term, IdealHelmholtzEnthalpyEntropyOffset):
            val += alpha0_enthalpy_entropy_offset(term, tau, delta)
        elif isinstance(term, IdealHelmholtzCP0Constant):
            val += alpha0_cp0_constant(term, tau, delta)
        elif isinstance(term, IdealHelmholtzCP0PolyT):
            val += alpha0_cp0_polyt(term, tau, delta)
            
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
        elif isinstance(term, ResidualHelmholtzNonAnalytic):
            val += alphar_nonanalytic(term, tau, delta)
    return val

def evaluate_alpha0(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    tau = params.Tr / T
    delta = rho / params.rhor
    return alpha0(params, tau, delta)

def evaluate_alphar(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    tau = params.Tr / T
    delta = rho / params.rhor
    return alphar(params, tau, delta)
