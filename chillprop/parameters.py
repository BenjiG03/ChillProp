import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Union, Optional

class IdealHelmholtzTerm(eqx.Module):
    pass

class IdealHelmholtzLead(IdealHelmholtzTerm):
    a1: float
    a2: float

class IdealHelmholtzLogTau(IdealHelmholtzTerm):
    a: float

class IdealHelmholtzEnthalpyEntropyOffset(IdealHelmholtzTerm):
    a1: float
    a2: float
    reference: str

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

class IdealHelmholtzPlanckEinsteinGeneralized(IdealHelmholtzTerm):
    n: jax.Array
    t: jax.Array
    c: jax.Array
    d: jax.Array

class IdealHelmholtzCP0Constant(IdealHelmholtzTerm):
    cp_over_R: float
    T0: float
    Tc: float
    t: jnp.ndarray # Dummy to satisfy array interface slightly? No.

class IdealHelmholtzCP0PolyT(IdealHelmholtzTerm):
    c: jnp.ndarray
    t: jnp.ndarray
    T0: float
    Tc: float

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
    using_tau_r: bool = False
class ViscosityRainwaterFriend(eqx.Module):
    b: jax.Array
    t: jax.Array

class ViscosityPowersOfTr(eqx.Module):
    a: jax.Array
    t: jax.Array
    T_reducing: float

class ViscosityFrictionTheory(eqx.Module):
    # Coefficients for friction theory
    Aa: jax.Array
    Aaa: jax.Array
    Ai: jax.Array
    Aii: jax.Array
    Ar: jax.Array
    Arr: jax.Array
    
    # Exponents
    Na: int
    Naa: int
    Ni: int
    Nii: int
    Nr: int
    Nrr: int
    
    # Constants
    T_reduce: float
    c1: float
    c2: float
    
    # Optional last
    Adrdr: Optional[jax.Array] = None

class ViscosityDiluteCollisionIntegral(eqx.Module):
    a: jax.Array
    t: jax.Array
    molar_mass: float
    C: float

class ViscosityInitialDensityEmpirical(eqx.Module):
    n: jax.Array
    d: jax.Array
    t: jax.Array
    T_reducing: float
    rhomolar_reducing: float

class ViscosityModifiedBatschinskiHildebrand(eqx.Module):
    a: jax.Array
    d1: jax.Array
    t1: jax.Array
    gamma: jax.Array
    l: jax.Array
    f: jax.Array
    d2: jax.Array
    t2: jax.Array
    g: jax.Array
    h: jax.Array
    p: jax.Array
    q: jax.Array
    T_reduce: float
    rhomolar_reduce: float

class ViscosityParameters(eqx.Module):
    dilute: Union[ViscosityPowersOfTr, ViscosityDiluteCollisionIntegral, dict]
    initial_density: Optional[Union[ViscosityRainwaterFriend, ViscosityInitialDensityEmpirical]]
    higher_order: Optional[Union[ViscosityFrictionTheory, ViscosityModifiedBatschinskiHildebrand, dict]]
    epsilon_over_k: float
    sigma_eta: float

class ConductivityRatioOfPolynomials(eqx.Module):
    A: jax.Array
    B: jax.Array
    n: jax.Array
    m: jax.Array
    T_reducing: float

class ConductivityDiluteEta0AndPoly(eqx.Module):
    A: jax.Array
    t: jax.Array

class ConductivityResidualPolynomialAndExponential(eqx.Module):
    A: jax.Array
    d: jax.Array
    t: jax.Array
    l: jax.Array
    gamma: jax.Array
    T_reducing: float
    rhomolar_reducing: float

class ConductivitySimplifiedOlchowySengers(eqx.Module):
    k: float
    R0: float
    nu: float
    gamma: float
    GAMMA: float
    zeta0: float
    qD: float
    T_ref: float

class ConductivityDiluteCO2HuberJPCRD2016(eqx.Module):
    pass

class ConductivityParameters(eqx.Module):
    dilute: Union[ConductivityRatioOfPolynomials, ConductivityDiluteEta0AndPoly, dict]
    residual: Optional[Union[ConductivityResidualPolynomialAndExponential, dict]]
    critical: Optional[Union[ConductivitySimplifiedOlchowySengers, dict]]

class FluidParameters(eqx.Module):
    name: str
    Tc: float
    rhoc: float
    Pc: float
    Tr: float
    rhor: float
    R: float
    M: float
    pseudo_pure: bool
    alpha0: List[IdealHelmholtzTerm]
    alphar: List[ResidualHelmholtzTerm]
    ancillary_pS: Optional[AncillaryEquation] = None
    ancillary_pL: Optional[AncillaryEquation] = None
    ancillary_pV: Optional[AncillaryEquation] = None
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
                n_arr = jnp.array(term['n'])
                t_arr = jnp.array(term['t'])
                if fluid.get('INFO', {}).get('CAS') == 'AIR.PPF':
                    pass
                alpha0.append(IdealHelmholtzPower(n_arr, t_arr))
            elif t_type == 'IdealGasHelmholtzPlanckEinstein':
                 alpha0.append(IdealHelmholtzPlanckEinstein(jnp.array(term['n']), jnp.array(term['t'])))
            elif t_type == 'IdealGasHelmholtzPlanckEinsteinFunctionT':
                 alpha0.append(IdealHelmholtzPlanckEinsteinFunctionT(
                     jnp.array(term['n']), 
                     jnp.array(term['v']),
                     float(term.get('Tcrit', fluid['STATES']['critical']['T'])) # fallback to fluid Tc
                 ))
            elif t_type == 'IdealGasHelmholtzPlanckEinsteinGeneralized':
                 alpha0.append(IdealHelmholtzPlanckEinsteinGeneralized(
                     jnp.array(term['n']),
                     jnp.array(term['t']),
                     jnp.array(term['c']),
                     jnp.array(term['d'])
                 ))
            elif t_type == 'IdealGasHelmholtzEnthalpyEntropyOffset':
                alpha0.append(IdealHelmholtzEnthalpyEntropyOffset(
                    term['a1'], 
                    term['a2'], 
                    term.get('reference', 'unknown')
                ))
            elif t_type == 'IdealHelmholtzPlanckEinsteinCP': # Handling potential alias
                 alpha0.append(IdealHelmholtzPlanckEinstein(jnp.array(term['n']), jnp.array(term['t'])))
            elif t_type == 'IdealGasHelmholtzCP0Constant':
                 alpha0.append(IdealHelmholtzCP0Constant(
                     float(term['cp_over_R']),
                     float(term['T0']),
                     float(term['Tc']),
                     jnp.array([])
                 ))
            elif t_type == 'IdealGasHelmholtzCP0PolyT':
                 alpha0.append(IdealHelmholtzCP0PolyT(
                     jnp.array(term['c']),
                     jnp.array(term['t']),
                     float(term['T0']),
                     float(term['Tc'])
                 ))

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
        
        # Critical and Reducing points
        crit = fluid['STATES']['critical']
        reducing = eos['STATES']['reducing']
        
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
                type=b['type'],
                using_tau_r=bool(b.get('using_tau_r', False))
            )

        # Transport
        trans = fluid.get('TRANSPORT', {})
        
        return cls(
            name=fluid['INFO']['NAME'],
            Tc=float(crit['T']),
            rhoc=float(crit['rhomolar']),
            Pc=float(crit['p']),
            Tr=float(reducing['T']),
            rhor=float(reducing['rhomolar']),
            R=float(eos['gas_constant']),
            M=float(eos['molar_mass']),
            pseudo_pure=bool(eos.get('pseudo_pure', False)),
            alpha0=alpha0,
            alphar=alphar,
            ancillary_pS=parse_anc('pS'),
            ancillary_pL=parse_anc('pL'),
            ancillary_pV=parse_anc('pV'),
            ancillary_rhoL=parse_anc('rhoL'),
            ancillary_rhoV=parse_anc('rhoV'),
            viscosity=cls._parse_viscosity(trans.get('viscosity')),
            conductivity=cls._parse_conductivity(trans.get('conductivity'), float(crit['T']))
        )


    @staticmethod
    def _parse_viscosity(data):
        if not data: return None
        # Handle list of viscosity models (pick first)
        if isinstance(data, list):
            data = data[0]
            
        dilute_data = data.get('dilute', {})
        dilute = None
        dtype = dilute_data.get('type')
        if dtype == 'powers_of_Tr':
            dilute = ViscosityPowersOfTr(
                jnp.array(dilute_data['a']),
                jnp.array(dilute_data['t']),
                float(dilute_data['T_reducing'])
            )
        elif dtype == 'collision_integral':
            dilute = ViscosityDiluteCollisionIntegral(
                a=jnp.array(dilute_data['a']),
                t=jnp.array(dilute_data['t']),
                molar_mass=float(dilute_data['molar_mass']),
                C=float(dilute_data['C'])
            )
        else:
            dilute = dilute_data.copy()
            for k in ['a', 't']:
                if k in dilute: dilute[k] = jnp.array(dilute[k])
            
        initial_density = None
        if 'initial_density' in data and data['initial_density']:
            init_data = data['initial_density']
            if init_data.get('type') == 'Rainwater-Friend':
                initial_density = ViscosityRainwaterFriend(
                    jnp.array(init_data['b']),
                    jnp.array(init_data['t'])
                )
            elif init_data.get('type') == 'empirical':
                initial_density = ViscosityInitialDensityEmpirical(
                    n=jnp.array(init_data['n']),
                    d=jnp.array(init_data['d']),
                    t=jnp.array(init_data['t']),
                    T_reducing=float(init_data.get('T_reducing', 1.0)),
                    rhomolar_reducing=float(init_data.get('rhomolar_reducing', 1.0))
                )
        
        ho = data.get('higher_order', {})
        higher_order = None
        if ho:
            if ho.get('type') == 'friction_theory':
                def get_arr(key):
                    return jnp.array(ho.get(key, [0.0, 0.0, 0.0]))
                
                higher_order = ViscosityFrictionTheory(
                    Aa=get_arr('Aa'), Aaa=get_arr('Aaa'), 
                    Ai=get_arr('Ai'), Aii=get_arr('Aii'),
                    Ar=get_arr('Ar'), Arr=get_arr('Arr'),
                    Na=int(ho.get('Na', 0)), Naa=int(ho.get('Naa', 0)),
                    Ni=int(ho.get('Ni', 0)), Nii=int(ho.get('Nii', 0)),
                    Nr=int(ho.get('Nr', 0)), Nrr=int(ho.get('Nrr', 0)),
                    T_reduce=float(ho.get('T_reduce', 1.0)),
                    c1=float(ho.get('c1', 0.0)),
                    c2=float(ho.get('c2', 0.0)),
                    Adrdr=jnp.array(ho.get('Adrdr')) if 'Adrdr' in ho else None
                )
            elif ho.get('type') == 'modified_Batschinski_Hildebrand':
                higher_order = ViscosityModifiedBatschinskiHildebrand(
                    a=jnp.array(ho.get('a', [])),
                    d1=jnp.array(ho.get('d1', [])),
                    t1=jnp.array(ho.get('t1', [])),
                    gamma=jnp.array(ho.get('gamma', [])),
                    l=jnp.array(ho.get('l', [])),
                    f=jnp.array(ho.get('f', [])),
                    d2=jnp.array(ho.get('d2', [])),
                    t2=jnp.array(ho.get('t2', [])),
                    g=jnp.array(ho.get('g', [])),
                    h=jnp.array(ho.get('h', [])),
                    p=jnp.array(ho.get('p', [])),
                    q=jnp.array(ho.get('q', [])),
                    T_reduce=float(ho.get('T_reduce', 1.0)),
                    rhomolar_reduce=float(ho.get('rhomolar_reduce', 1.0))
                )
            else:
                higher_order = ho # Fallback for dict
        
        return ViscosityParameters(
            dilute, 
            initial_density,
            higher_order,
            float(data.get('epsilon_over_k', 0.0)),
            float(data.get('sigma_eta', 0.0))
        )

    @staticmethod
    def _parse_conductivity(data, Tc: float):
        if not data: return None
        # Handle list of conductivity models (pick first)
        if isinstance(data, list):
            data = data[0]
            
        dilute_data = data.get('dilute', {})
        dilute = None
        dtype = dilute_data.get('type')
        if dtype == 'ratio_of_polynomials':
            dilute = ConductivityRatioOfPolynomials(
                jnp.array(dilute_data['A']),
                jnp.array(dilute_data['B']),
                jnp.array(dilute_data['n']),
                jnp.array(dilute_data['m']),
                float(dilute_data.get('T_reducing', 1.0))
            )
        elif dtype == 'eta0_and_poly':
            dilute = ConductivityDiluteEta0AndPoly(
                A=jnp.array(dilute_data['A']),
                t=jnp.array(dilute_data['t'])
            )
        elif dtype == 'CarbonDioxideHuberJPCRD2016':
            dilute = ConductivityDiluteCO2HuberJPCRD2016()
        else:
            dilute = dilute_data.copy()
            if 'A' in dilute: dilute['A'] = jnp.array(dilute['A'])
            if 't' in dilute: dilute['t'] = jnp.array(dilute['t'])
        
        res_data = data.get('residual', {})
        residual = None
        rtype = res_data.get('type')
        if rtype in ['polynomial', 'polynomial_and_exponential']:
            val_A = res_data.get('A')
            val_B = res_data.get('B')
            
            # Use B if present, otherwise A
            coeffs = val_B if val_B is not None else val_A
            if coeffs is None: coeffs = []
            
            A = jnp.array(coeffs)
            residual = ConductivityResidualPolynomialAndExponential(
                A=A,
                d=jnp.array(res_data['d']),
                t=jnp.array(res_data['t']),
                l=jnp.array(res_data.get('l', jnp.zeros(A.shape))),
                gamma=jnp.array(res_data.get('gamma', jnp.zeros(A.shape))),
                T_reducing=float(res_data.get('T_reducing', 0.0)), # 0.0 means use Tc
                rhomolar_reducing=float(res_data.get('rhomolar_reducing', 0.0))
            )
        else:
            residual = res_data.copy()
            for k in ['A', 'd', 't', 'l', 'gamma']:
                if k in residual: residual[k] = jnp.array(residual[k])
            
        crit_data = data.get('critical', {})
        critical = None
        if crit_data and crit_data.get('type') == 'simplified_Olchowy_Sengers':
            critical = ConductivitySimplifiedOlchowySengers(
                k=float(crit_data.get('k', 1.38064852e-23)),
                R0=float(crit_data.get('R0', 1.0)),
                nu=float(crit_data.get('nu', 0.63)),
                gamma=float(crit_data.get('gamma', 1.239)),
                GAMMA=float(crit_data.get('GAMMA', 0.0496)),
                zeta0=float(crit_data.get('zeta0', 1.9e-10)),
                qD=float(crit_data.get('qD', 2000000000.0)),
                T_ref=float(crit_data.get('T_ref', 1.5 * Tc))
            )
        else:
            critical = crit_data
        
        return ConductivityParameters(dilute, residual, critical)

