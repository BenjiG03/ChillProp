import equinox as eqx
import jax.numpy as jnp
from typing import List, Union, Optional

class IdealHelmholtzTerm(eqx.Module):
    pass

class IdealHelmholtzLead(IdealHelmholtzTerm):
    a1: float
    a2: float

class IdealHelmholtzLogTau(IdealHelmholtzTerm):
    a: float

class IdealHelmholtzPower(IdealHelmholtzTerm):
    n: jnp.ndarray
    t: jnp.ndarray

class IdealHelmholtzPlanckEinstein(IdealHelmholtzTerm):
    n: jnp.ndarray
    t: jnp.ndarray

class IdealHelmholtzPlanckEinsteinFunctionT(IdealHelmholtzTerm):
    n: jnp.ndarray
    v: jnp.ndarray
    Tcrit: float

class ResidualHelmholtzTerm(eqx.Module):
    pass

class ResidualHelmholtzPower(ResidualHelmholtzTerm):
    n: jnp.ndarray
    d: jnp.ndarray
    t: jnp.ndarray
    l: jnp.ndarray

class ResidualHelmholtzGaussian(ResidualHelmholtzTerm):
    n: jnp.ndarray
    d: jnp.ndarray
    t: jnp.ndarray
    eta: jnp.ndarray
    epsilon: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

class AncillaryEquation(eqx.Module):
    n: jnp.ndarray
    t: jnp.ndarray
    reducing_value: float
    T_r: float
    type: str

class ViscosityParameters(eqx.Module):
    # For now, we store as dict/leaves for JAX
    dilute: dict
    higher_order: Optional[dict] = None
    epsilon_over_k: float = 0.0
    sigma_eta: float = 0.0

class ConductivityParameters(eqx.Module):
    dilute: dict
    residual: dict
    critical: Optional[dict] = None

class FluidParameters(eqx.Module):
    name: str
    Tc: float
    rhoc: float
    Pc: float
    R: float
    M: float
    alpha0: List[IdealHelmholtzTerm]
    alphar: List[ResidualHelmholtzTerm]
    ancillary_p: Optional[AncillaryEquation] = None
    ancillary_rhoL: Optional[AncillaryEquation] = None
    ancillary_rhoV: Optional[AncillaryEquation] = None
    viscosity: Optional[ViscosityParameters] = None
    conductivity: Optional[ConductivityParameters] = None

    @classmethod
    def from_json(cls, data: dict):
        # Handle list vs dict
        fluid = data[0] if isinstance(data, list) else data
        
        # Access the first EOS formulation
        if 'EOS' not in fluid or not fluid['EOS']:
            raise ValueError("No EOS found in fluid data")
        
        eos = fluid['EOS'][0]
        
        # Parse Ideal Terms
        alpha0 = []
        for term in eos.get('alpha0', []):
            t_type = term.get('type')
            if t_type == 'IdealGasHelmholtzLead':
                alpha0.append(IdealHelmholtzLead(term['a1'], term['a2']))
            elif t_type == 'IdealGasHelmholtzLogTau':
                alpha0.append(IdealHelmholtzLogTau(term['a']))
            elif t_type == 'IdealGasHelmholtzPower':
                alpha0.append(IdealHelmholtzPower(jnp.array(term['n']), jnp.array(term['t'])))
            elif t_type == 'IdealGasHelmholtzPlanckEinstein':
                 alpha0.append(IdealHelmholtzPlanckEinstein(jnp.array(term['n']), jnp.array(term['t'])))
            elif t_type == 'IdealGasHelmholtzPlanckEinsteinFunctionT':
                 alpha0.append(IdealHelmholtzPlanckEinsteinFunctionT(
                     jnp.array(term['n']), 
                     jnp.array(term['v']),
                     float(term.get('Tcrit', fluid['STATES']['critical']['T'])) # fallback to fluid Tc
                 ))
            elif t_type == 'IdealHelmholtzPlanckEinsteinCP': # Handling potential alias
                 alpha0.append(IdealHelmholtzPlanckEinstein(jnp.array(term['n']), jnp.array(term['t'])))

        # Parse Residual Terms
        alphar = []
        for term in eos.get('alphar', []):
            t_type = term.get('type')
            if t_type == 'ResidualHelmholtzPower':
                l_arr = jnp.array(term['l']) if 'l' in term else jnp.zeros(len(term['n']))
                alphar.append(ResidualHelmholtzPower(
                    jnp.array(term['n']),
                    jnp.array(term['d']),
                    jnp.array(term['t']),
                    l_arr
                ))
            elif t_type == 'ResidualHelmholtzGaussian':
                 alphar.append(ResidualHelmholtzGaussian(
                    jnp.array(term['n']),
                    jnp.array(term['d']),
                    jnp.array(term['t']),
                    jnp.array(term['eta']),
                    jnp.array(term['epsilon']),
                    jnp.array(term['beta']),
                    jnp.array(term['gamma'])
                ))
        
        # Critical points
        crit = fluid['STATES']['critical']
        
        # Ancillaries
        anc = fluid.get('ANCILLARIES', {})
        
        def parse_anc(block_name):
            if block_name not in anc: return None
            b = anc[block_name]
            return AncillaryEquation(
                n=jnp.array(b['n']),
                t=jnp.array(b['t']),
                reducing_value=float(b['reducing_value']),
                T_r=float(b['T_r']),
                type=b['type']
            )

        # Transport
        trans = fluid.get('TRANSPORT', {})
        
        return cls(
            name=fluid['INFO']['NAME'],
            Tc=float(crit['T']),
            rhoc=float(crit['rhomolar']),
            Pc=float(crit['p']),
            R=float(eos['gas_constant']),
            M=float(eos['molar_mass']),
            alpha0=alpha0,
            alphar=alphar,
            ancillary_p=parse_anc('pS'),
            ancillary_rhoL=parse_anc('rhoL'),
            ancillary_rhoV=parse_anc('rhoV'),
            viscosity=cls._parse_viscosity(trans.get('viscosity')),
            conductivity=cls._parse_conductivity(trans.get('conductivity'))
        )


    @staticmethod
    def _parse_viscosity(data):
        if not data: return None
        dilute = data.get('dilute', {}).copy()
        for k in ['a', 't']:
            if k in dilute: dilute[k] = jnp.array(dilute[k])
        
        ho = data.get('higher_order', {})
        if ho:
            ho = ho.copy()
            for k in ['a','d1','d2','t1','t2','f','g','gamma','h','l','p','q']:
                if k in ho: ho[k] = jnp.array(ho[k])
        
        return ViscosityParameters(
            dilute, 
            ho if ho else None,
            float(data.get('epsilon_over_k', 0.0)),
            float(data.get('sigma_eta', 0.0))
        )

    @staticmethod
    def _parse_conductivity(data):
        if not data: return None
        dilute = data.get('dilute', {}).copy()
        if 'A' in dilute: dilute['A'] = jnp.array(dilute['A'])
        if 't' in dilute: dilute['t'] = jnp.array(dilute['t'])
        
        res = data.get('residual', {}).copy()
        for k in ['A', 'd', 't', 'l', 'gamma']:
            if k in res: res[k] = jnp.array(res[k])
            
        crit = data.get('critical', {})
        if crit: crit = crit.copy() # Leave as dict for now
        
        return ConductivityParameters(dilute, res, crit if crit else None)

