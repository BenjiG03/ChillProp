import jax
import jax.numpy as jnp
from typing import List, Union, Optional
from chillprop.parameters import (
    FluidParameters, ConductivityParameters, ViscosityParameters,
    ConductivityRatioOfPolynomials, ConductivityDiluteEta0AndPoly,
    ConductivityResidualPolynomialAndExponential, ConductivitySimplifiedOlchowySengers,
    ViscosityPowersOfTr, ViscosityDiluteCollisionIntegral,
    ViscosityRainwaterFriend, ViscosityInitialDensityEmpirical,
    ViscosityFrictionTheory, ViscosityModifiedBatschinskiHildebrand,
    ConductivityDiluteCO2HuberJPCRD2016
)
from chillprop import core

def viscosity_dilute(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Dilute viscosity [Pa-s]"""
    v = params.viscosity
    if v is None:
        return jnp.nan
    
    d = v.dilute
    if isinstance(d, ViscosityPowersOfTr):
        Tr = T / d.T_reducing
        return jnp.sum(d.a * (Tr ** d.t))
    elif isinstance(d, ViscosityDiluteCollisionIntegral):
        # η = C * sqrt(M_kgkmol * T) / (sigma_nm^2 * exp(sum(a_i * (ln(T*))^t_i)))
        # Match CoolProp/src/Backends/Helmholtz/TransportRoutines.cpp
        T_star = T / v.epsilon_over_k
        lnT_star = jnp.log(T_star)
        summer = jnp.sum(d.a * (lnT_star ** d.t))
        S = jnp.exp(summer)
        
        sigma_nm = v.sigma_eta * 1e9
        molar_mass_kgkmol = d.molar_mass * 1000.0
        
        return d.C * jnp.sqrt(molar_mass_kgkmol * T) / (sigma_nm**2 * S)
    elif isinstance(d, dict) and 'hardcoded' in d:
        if d.get('hardcoded') == 'CarbonDioxideLaeseckeJPCRD2017':
            Tr = T / 251.196
            a = jnp.array([0.235156, 12.90844, 1.635569, 0.4485661, 1.1578282e-2, 0.19708305e-2])
            t = jnp.array([-1, 0, 1, 2, 3, 4])
            return jnp.sum(a * (Tr**t)) * 1e-6
    elif isinstance(d, dict) and 'a' in d and 't' in d:
        # Generic power sum
        Tr = T / d.get('T_reducing', params.Tc)
        return jnp.sum(jnp.array(d['a']) * (Tr ** jnp.array(d['t'])))
    
    return jnp.nan

def viscosity_initial_density(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Initial density contribution to viscosity [Pa-s]"""
    v = params.viscosity
    if v is None or v.initial_density is None:
        return 0.0
    
    id = v.initial_density
    if isinstance(id, ViscosityRainwaterFriend):
        # CoolProp/src/Backends/Helmholtz/TransportRoutines.cpp:138
        # Uses Tstar and sigma
        Tstar = T / v.epsilon_over_k
        B_star = jnp.sum(id.b * (Tstar ** id.t))
        
        sigma = v.sigma_eta # [m]
        NA = 6.02214076e23
        B_eta = NA * (sigma**3) * B_star # [m^3/mol]
        
        # Contribution is eta_dilute * B_eta * rho
        eta0 = viscosity_dilute(params, T)
        return eta0 * B_eta * rho
    elif isinstance(id, ViscosityInitialDensityEmpirical):
        delta = rho / id.rhomolar_reducing
        tau = id.T_reducing / T
        B = jnp.sum(id.n * (delta ** id.d) * (tau ** id.t))
        return B
    return 0.0

def viscosity_higher_order(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Higher order contribution to viscosity [Pa-s]"""
    v = params.viscosity
    if v is None or v.higher_order is None:
        return 0.0
    
    ho = v.higher_order
    if isinstance(ho, ViscosityFrictionTheory):
        # Friction Theory
        R_u = 8.314462618 # Universal
        P_id = rho * R_u * T / params.M
        # Actually P_ref logic is complex in FT. Use highlevel PropSI for P_ref or similar.
        # But this requires solver.
        # FT is rarely used in core fluids like Air.
        return 0.0 
    elif isinstance(ho, ViscosityModifiedBatschinskiHildebrand):
        delta = rho / ho.rhomolar_reduce
        tau = ho.T_reduce / T
        
        S = jnp.sum(ho.a * (delta ** ho.d1) * (tau ** ho.t1) * jnp.exp(ho.gamma * (delta ** ho.l)))
        F = jnp.sum(ho.f * (delta ** ho.d2) * (tau ** ho.t2))
        delta0 = jnp.sum(ho.g * (tau ** ho.h)) / jnp.sum(ho.p * (tau ** ho.q))
        
        return S + F * (1.0 / (delta0 - delta) - 1.0 / delta0)
    elif isinstance(ho, dict) and 'hardcoded' in ho:
        hc = ho['hardcoded']
        if hc == 'CarbonDioxideLaeseckeJPCRD2017':
            # Eq. (9) from Laesecke, JPCRD, 2017
            Tt = 216.592
            rho_tL = 1178.53
            R = 188.9241525
            M = 44.0095e-3 # kg/mol? Wait params.M is kg/mol.
            # CoolProp implementation hardcodes 84446887.43579945 
            # eta_tL = pow(rho_tL, 2.0 / 3.0) * sqrt(HEOS.gas_constant() * Tt) / (pow(HEOS.molar_mass(), 1.0 / 6.0) * 84446887.43579945);
            
            # Use param values
            # Need to be careful with units.
            # HEOS.rhomass()
            rhomass = rho * params.M
            
            Tr = T / Tt
            rhor = rhomass / rho_tL
            
            # Hardcoded constant in CoolProp seems specific
            # Let's try to match exactly
            denom = (params.M**(1.0/6.0)) * 84446887.43579945
            eta_tL = (rho_tL**(2.0/3.0)) * jnp.sqrt(params.R * Tt) / denom
            
            c1 = 0.360603235428487
            c2 = 0.121550806591497
            gamma = 8.06282737481277
            
            residual = eta_tL * (c1 * Tr * (rhor**3) + ((rhor**2) + (rhor**gamma)) / (Tr - c2))
            return residual
            
        elif hc == 'Hydrogen':
            # CoolProp/src/Backends/Helmholtz/TransportRoutines.cpp:308
            Tr = T / 33.145
            rhomass = rho * params.M
            rhor = rhomass * 0.011 # Note: weird normalization in CoolProp source
            
            c = jnp.array([0, 6.43449673e-6, 4.56334068e-2, 2.32797868e-1, 9.58326120e-1, 1.27941189e-1, 3.63576595e-1])
            
            term = c[2]*Tr + c[3]/Tr + c[4]*(rhor**2)/(c[5] + Tr) + c[6]*(rhor**6)
            return c[1] * (rhor**2) * jnp.exp(term)
    
    return 0.0

def viscosity(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Viscosity [Pa-s]"""
    eta0 = viscosity_dilute(params, T)
    eta_initial = viscosity_initial_density(params, rho, T)
    eta_ho = viscosity_higher_order(params, rho, T)
    return eta0 + eta_initial + eta_ho

def _calc_cv0_r(params: FluidParameters, T: jax.Array) -> jax.Array:
    # Calculate Ideal Gas Heat Capacity / R
    from chillprop import heos
    tau = params.Tr / T
    delta_dummy = jnp.ones_like(tau)
    
    def alpha0_tau(t):
        return heos.alpha0(params, t, delta_dummy)
    
    # Second derivative w.r.t. tau
    if len(tau.shape) == 0:
        d2 = jax.grad(jax.grad(alpha0_tau))(tau)
    else:
        d2 = jax.vmap(jax.grad(jax.grad(alpha0_tau)))(tau)
        
    return -(tau**2) * d2

def thermal_conductivity_dilute(params: FluidParameters, T: jax.Array) -> jax.Array:
    """Dilute thermal conductivity [W/m/K]"""
    c = params.conductivity
    if c is None: return 0.0
    
    d = c.dilute
    if isinstance(d, ConductivityRatioOfPolynomials):
        Tr = T / d.T_reducing
        num = jnp.sum(d.A * (Tr ** d.n))
        den = jnp.sum(d.B * (Tr ** d.m))
        return num / den
    elif isinstance(d, ConductivityDiluteEta0AndPoly):
        eta0_uPas = viscosity_dilute(params, T) * 1e6
        lambda_val = d.A[0] * eta0_uPas
        if d.A.shape[0] > 1:
            tau = params.Tr / T
            lambda_val += jnp.sum(d.A[1:] * (tau ** d.t[1:]))
        return lambda_val

    elif isinstance(d, ConductivityDiluteCO2HuberJPCRD2016):
        # Huber 2016 Eq. (3)
        tau = params.Tr / T
        l = jnp.array([0.0151874307, 0.0280674040, 0.0228564190, -0.00741624210])
        # lambda_0 = pow(tau, -0.5) / (l[0] + l[1] * tau + l[2] * pow(tau, 2) + l[3] * pow(tau, 3));  // [mW/m/K]
        # return lambda_0 / 1000;
        
        denom = l[0] + l[1]*tau + l[2]*(tau**2) + l[3]*(tau**3)
        lambda_0_mW = (tau**(-0.5)) / denom
        return lambda_0_mW / 1000.0
    elif isinstance(d, dict) and d.get('type') == 'kinetic_theory':
        # Modified Eucken correlation
        eta0 = viscosity_dilute(params, T)
        cv0_r = _calc_cv0_r(params, T)
        f_trans = 2.5
        f_int = 1.32
        cv_trans_r = 1.5
        cv_int_r = cv0_r - 1.5
        term = f_trans * cv_trans_r + f_int * cv_int_r
        return eta0 * term * params.R / params.M
    return 0.0

def thermal_conductivity_residual(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Residual thermal conductivity [W/m/K]"""
    c = params.conductivity
    if c is None or c.residual is None: return 0.0
    
    res = c.residual
    if isinstance(res, ConductivityResidualPolynomialAndExponential):
        Tr_red = res.T_reducing if res.T_reducing > 0.0 else params.Tr
        rho_red = res.rhomolar_reducing if res.rhomolar_reducing > 0.0 else params.rhor
        
        delta = rho / rho_red
        tau = Tr_red / T
        
        return jnp.sum(res.A * (delta**res.d) * (tau**res.t) * jnp.exp(-res.gamma * (delta**res.l)))

    elif isinstance(res, dict) and res.get('type') == 'polynomial':
        # Handle dict fallback for polynomial if any
        Tr_red = float(res.get('T_reducing', params.Tc))
        rho_red = float(res.get('rhomolar_reducing', params.rhoc))
        delta = rho / rho_red
        tau = Tr_red / T
        
        coeffs = res.get('B', res.get('A'))
        if coeffs is None: return 0.0
        
        return jnp.sum(jnp.array(coeffs) * (delta**jnp.array(res['d'])) * (tau**jnp.array(res['t'])))
        
    return 0.0

def _dp_drho_T_delta(params: FluidParameters, T: jax.Array, delta: jax.Array) -> jax.Array:
    from chillprop.heos import alphar
    tau = params.Tc / T
    def ar_delta(d):
        return alphar(params, tau, d)
    ar_d = jax.grad(ar_delta)(delta)
    ar_dd = jax.grad(jax.grad(ar_delta))(delta)
    return params.R * T * (1.0 + 2.0 * delta * ar_d + delta**2 * ar_dd)

def thermal_conductivity_critical(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Critical enhancement to thermal conductivity [W/m/K]"""
    c = params.conductivity
    if c is None or c.critical is None: return 0.0
    lc = c.critical
    
    if isinstance(lc, ConductivitySimplifiedOlchowySengers):
        delta = rho / params.rhoc
        dp_drho = _dp_drho_T_delta(params, T, delta)
        X = (params.Pc / params.rhoc**2) * rho / dp_drho
        
        tau_ref = params.Tc / lc.T_ref
        dp_drho_ref = _dp_drho_T_delta(params, lc.T_ref, delta)
        Xref = (params.Pc / params.rhoc**2) * rho / dp_drho_ref * (lc.T_ref / T)
        
        num = X - Xref
        # DBL_EPSILON * 10 ~ 2e-15
        if num < 2.22e-15:
            return 0.0
        
        zeta = lc.zeta0 * (num / lc.GAMMA)**(lc.nu / lc.gamma)
        
        cp = core.cpmolar(params, rho, T)
        cv = core.cvmolar(params, rho, T)
        mu = viscosity(params, rho, T)
        
        qd_zeta = lc.qD * zeta
        pi = jnp.pi
        
        omega_tilde = (2.0 / pi) * (((cp - cv) / cp) * jnp.arctan(qd_zeta) + (cv / cp) * qd_zeta)
        # OMEGA_tilde0 = 2.0 / pi * (1.0 - exp(-1.0 / (1.0 / (qD * zeta) + 1.0 / 3.0 * (zeta * qD) * (zeta * qD) / delta / delta)))
        omega_tilde0 = (2.0 / pi) * (1.0 - jnp.exp(-1.0 / (1.0 / qd_zeta + (1.0/3.0) * qd_zeta**2 / delta**2)))
        
        lambda_crit = (rho * cp * lc.R0 * lc.k * T) / (6.0 * pi * mu * zeta) * (omega_tilde - omega_tilde0)
        return lambda_crit
        
    return 0.0

def thermal_conductivity(params: FluidParameters, rho: jax.Array, T: jax.Array) -> jax.Array:
    """Thermal Conductivity [W/m/K]"""
    l0 = thermal_conductivity_dilute(params, T)
    lr = thermal_conductivity_residual(params, rho, T)
    lc = thermal_conductivity_critical(params, rho, T)
    return l0 + lr + lc
