import jax
import jax.numpy as jnp
import numpy as np
from chillprop.parameters import FluidParameters, ResidualHelmholtzPower, ResidualHelmholtzGaussian
from scripts.extract_params import extract_fluid_params
import CoolProp.CoolProp as CP

jax.config.update("jax_enable_x64", True)

def debug_heos():
    # Load Params
    data = extract_fluid_params('Nitrogen')
    params = FluidParameters.from_json(data)
    
    # State
    T = 151.4304
    rho = 11183.9015
    Tc = params.Tc
    rhoc = params.rhoc
    
    tau = Tc / T
    delta = rho / rhoc
    
    print(f"State: T={T}, rho={rho}")
    print(f"Tc={Tc}, rhoc={rhoc}")
    print(f"tau={tau}, delta={delta}")
    
    # Expected Z
    AS = CP.AbstractState("HEOS", "Nitrogen")
    AS.update(CP.DmolarT_INPUTS, rho, T)
    P_ref = AS.p()
    Z_ref = P_ref / (rho * params.R * T)
    print(f"Ref P={P_ref}, Z={Z_ref}")
    target_delta_alphar_delta = Z_ref - 1.0
    print(f"Target delta * alphar_delta = {target_delta_alphar_delta}")
    
    # Calculate Terms
    total_val = 0.0
    total_der = 0.0
    
    print("\n--- Residual Power Terms ---")
    for i, term in enumerate(params.alphar):
        if isinstance(term, ResidualHelmholtzPower):
            # We must iterate over the vector in the term
            # The term object contains arrays n, d, t, l
            # We implemented alphar_power to sum them all.
            # Let's break it down manually.
            
            ns = term.n
            ds = term.d
            ts = term.t
            ls = term.l if hasattr(term, 'l') else jnp.zeros_like(ns)
            
            for j in range(len(ns)):
                n, d, t, l = ns[j], ds[j], ts[j], ls[j]
                
                # Value
                exp_factor = np.exp(-delta**l) if l != 0 else 1.0
                val = n * (delta**d) * (tau**t) * exp_factor
                
                # Deriv w.r.t delta
                # d/ddelta = n * tau^t * [ d*delta^(d-1)*exp + delta^d*exp*(-l*delta^(l-1)) ]
                #          = n * tau^t * delta^(d-1) * exp * [ d - l*delta^l ]
                
                # delta * deriv = n * tau^t * delta^d * exp * [ d - l*delta^l ]
                #               = val * [ d - l*delta^l ]
                
                d_term = d - l * (delta**l)
                term_contribution = val * d_term
                
                total_val += val
                total_der += term_contribution
                
                print(f"Power Term {j}: n={n:.4f}, d={d}, t={t:.4f}, l={l} -> Val={val:.4f}, Z_contrib={term_contribution:.4f}")

        elif isinstance(term, ResidualHelmholtzGaussian):
            print("--- Residual Gaussian Terms ---")
            ns = term.n
            ds = term.d
            ts = term.t
            etas = term.eta
            epsilons = term.epsilon
            betas = term.beta
            gammas = term.gamma
            
            for j in range(len(ns)):
                n, d, t = ns[j], ds[j], ts[j]
                eta, eps, beta, gamma = etas[j], epsilons[j], betas[j], gammas[j]
                
                # Val
                # exp(-eta(delta-eps)^2 - beta(tau-gamma)^2)
                arg = -eta*(delta-eps)**2 - beta*(tau-gamma)**2
                exp_factor = np.exp(arg)
                val = n * delta**d * tau**t * exp_factor
                
                # Deriv
                # d/ddelta (val)
                # val * [ d/delta - 2*eta*(delta-eps) ]
                # delta * deriv = val * [ d - 2*eta*delta*(delta-eps) ]
                
                d_term = d - 2 * eta * delta * (delta - eps)
                term_contribution = val * d_term
                
                total_val += val
                total_der += term_contribution
                
                print(f"Gaussian Term {j}: n={n:.4f}, d={d}, -> Val={val:.4f}, Z_contrib={term_contribution:.4f}")

    print(f"\nTotal Z_contrib (delta * alphar_delta) = {total_der}")
    print(f"Calculated Z = {1.0 + total_der}")
    print(f"Mismatch: {1.0 + total_der - Z_ref}")

if __name__ == "__main__":
    debug_heos()
