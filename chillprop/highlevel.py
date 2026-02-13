import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union
from chillprop.parameters import FluidParameters
from chillprop.core import pressure, enthalpy, entropy, internal_energy, speed_sound
from chillprop.solver import solve_rho_PT, solve_rhoT_Ph, solve_rhoT_Ps
from chillprop.phases import solve_vle, vapor_quality, get_phase, rhol_anc, rhov_anc, evaluate_ancillary
from chillprop.transport import viscosity, thermal_conductivity
import CoolProp.CoolProp as CP
from scripts.extract_params import extract_fluid_params

# Global cache for fluid parameters to avoid reloading
_FLUID_CACHE = {}

def get_params(fluid: str) -> FluidParameters:
    if fluid not in _FLUID_CACHE:
        data = extract_fluid_params(fluid)
        _FLUID_CACHE[fluid] = FluidParameters.from_json(data)
    return _FLUID_CACHE[fluid]

# String mapping for CoolProp compatibility
KEY_MAP = {
    'D': 'rho', 'Dmolar': 'rho', 'Density': 'rho',
    'P': 'P', 'Pressure': 'P',
    'T': 'T', 'Temperature': 'T',
    'H': 'h', 'Hmolar': 'h', 'Enthalpy': 'h',
    'S': 's', 'Smolar': 's', 'Entropy': 's',
    'U': 'u', 'Umolar': 'u', 'InternalEnergy': 'u',
    'Q': 'Q'
}

def PropsSI(out_key: str, key1: str, val1: float, key2: str, val2: float, fluid: str) -> float:
    """CoolProp-like PropsSI interface."""
    params = get_params(fluid)
    
    k1 = KEY_MAP.get(key1)
    k2 = KEY_MAP.get(key2)
    ok = KEY_MAP.get(out_key, out_key)
    
    # Simple dispatcher for input pairs
    # (T, P), (P, T), (P, H), (P, S)
    
    # Note: For JAX-friendliness, we'll want this to be JIT-able.
    # But PropsSI is often called with strings which JAX doesn't like in JIT.
    # We can separate the "parsed" logic from the "jitted" logic.
    
    return _PropsSI_internal(params, out_key, k1, val1, k2, val2)

def _PropsSI_internal(params, out_key, k1, v1, k2, v2):
    # Handle mass to molar input conversions
    # Map for easy lookup
    v1_molar = v1
    v2_molar = v2
    
    # Input mass-to-molar
    if k1 == 'rho' and out_key != 'Dmolar' and 'Density' not in out_key: # If 'Density' or 'D' is passed, assume mass? Wait.
        # Actually CoolProp distinguishes 'D' (kg/m3) and 'Dmolar' (mol/m3).
        pass
    
    # Let's check keys specifically
    if k1 == 'rho' and 'molar' not in k1: pass # Handle later
    
    # Simpler approach: check the keys passed to PropsSI from KEY_MAP
    # Actually, let's just use a set of keys that are mass-based in CoolProp.
    MASS_KEYS = ['D', 'Density', 'H', 'Enthalpy', 'S', 'Entropy', 'U', 'InternalEnergy']
    
    # Real logic:
    # If key is in MASS_KEYS, divide/multiply by molar mass M.
    
    # For now, let's just handle the basics:
    val1 = v1 / params.M if k1 in ['h', 's', 'u'] else v1 # Wait, h_mass = h_molar / M? No. 
    # Enthalpy: J/kg = J/mol / (kg/mol) -> h_mass = h_molar / M. Correct.
    # Density: kg/m^3 = mol/m^3 * kg/mol -> rho_mass = rho_molar * M. Correct.
    
    input_v1 = v1
    if k1 == 'rho' and 'molar' not in k1: input_v1 = v1 / params.M
    if k1 in ['h', 's', 'u']: input_v1 = v1 * params.M # J/kg * kg/mol = J/mol
    
    input_v2 = v2
    if k2 == 'rho' and 'molar' not in k2: input_v2 = v2 / params.M
    if k2 in ['h', 's', 'u']: input_v2 = v2 * params.M
    
    # Solver dispatch
    if (k1 == 'P' and k2 == 'T') or (k1 == 'T' and k2 == 'P'):
        P = input_v1 if k1 == 'P' else input_v2
        T = input_v2 if k1 == 'P' else input_v1
        rho = solve_rho_PT(params, jnp.array(P), jnp.array(T))
    elif (k1 == 'P' and k2 == 'h') or (k1 == 'h' and k2 == 'P'):
        P = input_v1 if k1 == 'P' else input_v2
        h = input_v2 if k1 == 'P' else input_v1
        res = solve_rhoT_Ph(params, jnp.array(P), jnp.array(h))
        rho, T = res[0], res[1]
    elif (k1 == 'P' and k2 == 's') or (k1 == 's' and k2 == 'P'):
        P = input_v1 if k1 == 'P' else input_v2
        s = input_v2 if k1 == 'P' else input_v1
        res = solve_rhoT_Ps(params, jnp.array(P), jnp.array(s))
        rho, T = res[0], res[1]
    elif k1 == 'T' and k2 == 'Q':
        T = input_v1
        Q = input_v2
        rho_l, rho_v = solve_vle(params, jnp.array(T))
        rho = 1.0 / (Q / rho_v + (1.0 - Q) / rho_l)
    elif (k1 == 'T' and k2 == 'rho') or (k1 == 'rho' and k2 == 'T'):
        T = input_v1 if k1 == 'T' else input_v2
        rho = input_v2 if k1 == 'T' else input_v1
    else:
        raise NotImplementedError(f"Input pair ({k1}, {k2}) not yet supported")


    # Determine if we are in two-phase region to weight properties
    # Note: T and rho are now known.
    rho_l, rho_v = solve_vle(params, T)
    # Use small epsilon for boundary checks
    is_twophase = (T < params.Tc) & (rho >= rho_v * 0.999) & (rho <= rho_l * 1.001)
    Q_calc = vapor_quality(rho, rho_l, rho_v)
    
    def get_prop(func_molar, mass_mult=1.0):
        if is_twophase:
            # For pseudo-pure, we must be careful with P
            if params.pseudo_pure:
                # Weighted P, H, S etc. between bubble and dew ancillaries
                pl = evaluate_ancillary(params.ancillary_pL, T) if params.ancillary_pL else pressure(params, rho_l, T)
                pv = evaluate_ancillary(params.ancillary_pV, T) if params.ancillary_pV else pressure(params, rho_v, T)
                val_l = func_molar(params, rho_l, T)
                val_v = func_molar(params, rho_v, T)
                # If the function is pressure, we use the weighted P
                if func_molar == pressure:
                    val = (1.0 - Q_calc) * pl + Q_calc * pv
                else:
                    val = (1.0 - Q_calc) * val_l + Q_calc * val_v
            else:
                val_l = func_molar(params, rho_l, T)
                val_v = func_molar(params, rho_v, T)
                val = (1.0 - Q) * val_l + Q * val_v
        else:
            val = func_molar(params, rho, T)
        return float(val * mass_mult)

    # Output selection and mass-to-molar conversion
    if out_key in ['D', 'Density']: return float(rho * params.M)
    if out_key in ['Dmolar']: return float(rho)
    if out_key in ['T', 'Temperature']: return float(T)
    if out_key in ['P', 'Pressure']: return get_prop(pressure) # P is same for both in VLE
    if out_key in ['H', 'Enthalpy']: return get_prop(enthalpy, 1.0/params.M)
    if out_key in ['Hmolar']: return get_prop(enthalpy)
    if out_key in ['S', 'Entropy']: return get_prop(entropy, 1.0/params.M)
    if out_key in ['Smolar']: return get_prop(entropy)
    if out_key in ['U', 'InternalEnergy']: return get_prop(internal_energy, 1.0/params.M)
    if out_key in ['Umolar']: return get_prop(internal_energy)
    if out_key == 'Q':
        return float(Q)
    if out_key in ['V', 'viscosity']: return float(viscosity(params, rho, T))
    if out_key in ['L', 'conductivity']: return float(thermal_conductivity(params, rho, T))
    if out_key in ['A', 'speed_sound']: return float(speed_sound(params, rho, T))
    
    raise ValueError(f"Output key {out_key} not supported")


class AbstractState:
    """CoolProp-like AbstractState wrapper."""
    def __init__(self, backend: str, fluid: str):
        self.params = get_params(fluid)
        self.rho = None
        self.T = None

    def update(self, input_pair: int, val1: float, val2: float):
        import CoolProp.CoolProp as CP
        
        if input_pair == CP.PT_INPUTS:
            self.rho = solve_rho_PT(self.params, jnp.array(val1), jnp.array(val2))
            self.T = jnp.array(val2)
        elif input_pair == CP.HmassP_INPUTS:
            h_molar = val1 * self.params.M
            res = solve_rhoT_Ph(self.params, jnp.array(val2), jnp.array(h_molar))
            self.rho, self.T = res[0], res[1]
        elif input_pair == CP.SmassP_INPUTS:
            s_molar = val1 * self.params.M
            res = solve_rhoT_Ps(self.params, jnp.array(val2), jnp.array(s_molar))
            self.rho, self.T = res[0], res[1]
        elif input_pair == CP.QT_INPUTS:
            self.T = jnp.array(val2)
            rho_l, rho_v = solve_vle(self.params, self.T)
            self.rho = 1.0 / (val1 / rho_v + (1.0 - val1) / rho_l)
        else:
            raise NotImplementedError(f"AbstractState.update for pair {input_pair} not implemented")

    def rhomolar(self): return float(self.rho)
    def rhomass(self): return float(self.rho * self.params.M)
    def hmolar(self): return float(enthalpy(self.params, self.rho, self.T))
    def hmass(self): return float(self.hmolar() / self.params.M)
    def smolar(self): return float(entropy(self.params, self.rho, self.T))
    def smass(self): return float(self.smolar() / self.params.M)
    def umolar(self): return float(internal_energy(self.params, self.rho, self.T))
    def umass(self): return float(self.umolar() / self.params.M)
    def p(self): return float(pressure(self.params, self.rho, self.T))
    def T(self): return float(self.T)
    def Q(self):
        rho_l, rho_v = solve_vle(self.params, self.T)
        return float(vapor_quality(self.rho, rho_l, rho_v))
    
    def keyed_output(self, key: int):
        import CoolProp.CoolProp as CP
        if key == CP.iDmolar: return self.rhomolar()
        if key == CP.iDmass: return self.rhomass()
        if key == CP.iT: return self.T()
        if key == CP.iP: return self.p()
        if key == CP.iHmolar: return self.hmolar()
        if key == CP.iHmass: return self.hmass()
        if key == CP.iSmolar: return self.smolar()
        if key == CP.iSmass: return self.smass()
        if key == CP.iUmolar: return self.umolar()
        if key == CP.iUmass: return self.umass()
        if key == CP.iQ: return self.Q()
        raise ValueError(f"Key {key} not supported")

