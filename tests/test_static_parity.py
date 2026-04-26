import pytest
import json
import numpy as np
import jax.numpy as jnp
import jax
from importlib import resources
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

# Enable double precision
jax.config.update("jax_enable_x64", True)

def test_nitrogen_static_parity():
    fluid_name = 'Nitrogen'
    with resources.files("chillprop").joinpath("data", f"{fluid_name}.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    
    # Parse into JAX structure
    params = FluidParameters.from_json(data)
    
    # Reference
    fluid = data[0] if isinstance(data, list) else data
    eos_data = fluid['EOS'][0]
    crit_data = fluid['STATES']['critical']
    
    # Assert base parameters
    assert params.Tc == crit_data['T']
    assert params.rhoc == crit_data['rhomolar']
    
    # Check Terms
    ref_alpha0 = eos_data.get('alpha0', [])
    assert len(params.alpha0) == len(ref_alpha0)
    
    for jax_term, ref_term in zip(params.alpha0, ref_alpha0):
        t_type = ref_term.get('type')
        if t_type == 'IdealGasHelmholtzLead':
            assert isinstance(jax_term, IdealHelmholtzLead)
            assert jnp.isclose(jax_term.a1, ref_term['a1'], atol=0.0)
            assert jnp.isclose(jax_term.a2, ref_term['a2'], atol=0.0)
        elif t_type == 'IdealGasHelmholtzLogTau':
             assert isinstance(jax_term, IdealHelmholtzLogTau)
             assert jnp.isclose(jax_term.a, ref_term['a'], atol=0.0)
        elif t_type == 'IdealGasHelmholtzPower':
            assert isinstance(jax_term, IdealHelmholtzPower)
            assert jnp.allclose(jax_term.n, jnp.array(ref_term['n']), atol=0.0)
            assert jnp.allclose(jax_term.t, jnp.array(ref_term['t']), atol=0.0)
        elif t_type == 'IdealGasHelmholtzPlanckEinsteinFunctionT':
            assert isinstance(jax_term, IdealHelmholtzPlanckEinsteinFunctionT)
            assert jnp.allclose(jax_term.n, jnp.array(ref_term['n']), atol=0.0)
            assert jnp.allclose(jax_term.v, jnp.array(ref_term['v']), atol=0.0)
            assert jnp.isclose(jax_term.Tcrit, ref_term.get('Tcrit', params.Tc), atol=0.0)

    # Check AlphaR
    ref_alphar = eos_data.get('alphar', [])
    assert len(params.alphar) == len(ref_alphar)
    
    for jax_term, ref_term in zip(params.alphar, ref_alphar):
        t_type = ref_term.get('type')
        if t_type == 'ResidualHelmholtzPower':
             assert isinstance(jax_term, ResidualHelmholtzPower)
             assert jnp.allclose(jax_term.n, jnp.array(ref_term['n']), atol=0.0)
             assert jnp.allclose(jax_term.d, jnp.array(ref_term['d']), atol=0.0)
             assert jnp.allclose(jax_term.t, jnp.array(ref_term['t']), atol=0.0)
             if 'l' in ref_term:
                 assert jnp.allclose(jax_term.l, jnp.array(ref_term['l']), atol=0.0)
        elif t_type == 'ResidualHelmholtzGaussian':
             assert isinstance(jax_term, ResidualHelmholtzGaussian)
             assert jnp.allclose(jax_term.n, jnp.array(ref_term['n']), atol=0.0)
             assert jnp.allclose(jax_term.d, jnp.array(ref_term['d']), atol=0.0)
             assert jnp.allclose(jax_term.t, jnp.array(ref_term['t']), atol=0.0)
             assert jnp.allclose(jax_term.eta, jnp.array(ref_term['eta']), atol=0.0)
             assert jnp.allclose(jax_term.epsilon, jnp.array(ref_term['epsilon']), atol=0.0)
             assert jnp.allclose(jax_term.beta, jnp.array(ref_term['beta']), atol=0.0)
             assert jnp.allclose(jax_term.gamma, jnp.array(ref_term['gamma']), atol=0.0)

if __name__ == "__main__":
    test_nitrogen_static_parity()
