from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chillprop.core import cpmolar, cvmolar, enthalpy, entropy, internal_energy, pressure, speed_sound
from chillprop.parameters import FluidParameters
from chillprop.phases import evaluate_ancillary, get_phase, rhol_anc, rhov_anc, solve_vle, vapor_quality
from chillprop.solver import solve_rho_PT, solve_rhoT_Ph, solve_rhoT_Ps
from chillprop.transport import thermal_conductivity, viscosity


_FLUID_CACHE: dict[str, FluidParameters] = {}
_ALIAS_CACHE: dict[str, str] = {}
_FLUID_DATA_CACHE: dict[str, dict[str, Any]] = {}

# Minimal CoolProp-compatible parameter and input-pair constants used by this repo.
iT = 19
iP = 20
iQ = 21
iDmolar = 24
iHmolar = 25
iSmolar = 26
iCpmolar = 27
iCvmolar = 29
iUmolar = 30
iDmass = 36
iHmass = 37
iSmass = 38
iCpmass = 39
iCvmass = 41
iUmass = 42
iviscosity = 51
iconductivity = 52
ispeed_sound = 54
iPrandtl = 53
iZ = 83
iPhase = 119
imolar_mass = 2
igas_constant = 1
iacentric_factor = 3
irhomolar_critical = 5
iT_critical = 7
irhomass_critical = 8
iP_critical = 9
iT_triple = 11
iP_triple = 12
iT_min = 13
iT_max = 14
iP_max = 15
iP_min = 16

iphase_liquid = 0
iphase_supercritical = 1
iphase_supercritical_gas = 2
iphase_supercritical_liquid = 3
iphase_critical_point = 4
iphase_gas = 5
iphase_twophase = 6
iphase_unknown = 7
iphase_not_imposed = 8

QT_INPUTS = 1
PQ_INPUTS = 2
QSmolar_INPUTS = 3
QSmass_INPUTS = 4
HmolarQ_INPUTS = 5
HmassQ_INPUTS = 6
DmolarQ_INPUTS = 7
DmassQ_INPUTS = 8
PT_INPUTS = 9
DmassT_INPUTS = 10
DmolarT_INPUTS = 11
HmolarT_INPUTS = 12
HmassT_INPUTS = 13
SmolarT_INPUTS = 14
SmassT_INPUTS = 15
TUmolar_INPUTS = 16
TUmass_INPUTS = 17
DmassP_INPUTS = 18
DmolarP_INPUTS = 19
HmassP_INPUTS = 20
HmolarP_INPUTS = 21
PSmass_INPUTS = 22
PSmolar_INPUTS = 23
PUmass_INPUTS = 24
PUmolar_INPUTS = 25
HmassSmass_INPUTS = 26
HmolarSmolar_INPUTS = 27
SmassUmass_INPUTS = 28
SmolarUmolar_INPUTS = 29
DmassHmass_INPUTS = 30
DmolarHmolar_INPUTS = 31
DmassSmass_INPUTS = 32
DmolarSmolar_INPUTS = 33
DmassUmass_INPUTS = 34
DmolarUmolar_INPUTS = 35

_PHASE_INDEX = {
    "phase_liquid": iphase_liquid,
    "phase_supercritical": iphase_supercritical,
    "phase_supercritical_gas": iphase_supercritical_gas,
    "phase_supercritical_liquid": iphase_supercritical_liquid,
    "phase_critical_point": iphase_critical_point,
    "phase_gas": iphase_gas,
    "phase_twophase": iphase_twophase,
    "phase_unknown": iphase_unknown,
    "phase_not_imposed": iphase_not_imposed,
    "iphase_liquid": iphase_liquid,
    "iphase_supercritical": iphase_supercritical,
    "iphase_supercritical_gas": iphase_supercritical_gas,
    "iphase_supercritical_liquid": iphase_supercritical_liquid,
    "iphase_critical_point": iphase_critical_point,
    "iphase_gas": iphase_gas,
    "iphase_twophase": iphase_twophase,
    "iphase_unknown": iphase_unknown,
    "iphase_not_imposed": iphase_not_imposed,
}

_DERIVATIVE_RE = re.compile(r"^d\((?P<of>[^)]+)\)/d\((?P<wrt>[^)]+)\)\|(?P<const>.+)$")
_SUPPORTED_BACKENDS = {"", "HEOS"}
_FLUID_ALIAS_OVERRIDES = {
    "n-propane": "Propane",
}
_TRIVIAL_KEYS = {
    "Tcrit",
    "T_critical",
    "pcrit",
    "Pcrit",
    "p_critical",
    "rhocrit",
    "rhomolar_critical",
    "rhomass_critical",
    "Ttriple",
    "T_triple",
    "ptriple",
    "p_triple",
    "Tmin",
    "T_min",
    "Tmax",
    "T_max",
    "Pmax",
    "pmax",
    "P_min",
    "Pmin",
    "pmin",
    "acentric",
    "acentric_factor",
    "molar_mass",
    "M",
    "MolarMass",
    "gas_constant",
    "R",
    "R_u",
}

_INPUT_ALIASES = {
    "T": ("T", "mass"),
    "Temperature": ("T", "mass"),
    "P": ("P", "mass"),
    "Pressure": ("P", "mass"),
    "Q": ("Q", "mass"),
    "D": ("Dmolar", "mass"),
    "Density": ("Dmolar", "mass"),
    "Dmolar": ("Dmolar", "molar"),
    "H": ("Hmolar", "mass"),
    "Enthalpy": ("Hmolar", "mass"),
    "Hmolar": ("Hmolar", "molar"),
    "S": ("Smolar", "mass"),
    "Entropy": ("Smolar", "mass"),
    "Smolar": ("Smolar", "molar"),
    "U": ("Umolar", "mass"),
    "InternalEnergy": ("Umolar", "mass"),
    "Umolar": ("Umolar", "molar"),
}


def _normalize_fluid_name(fluid: str) -> str:
    """Resolve a user fluid string to the canonical bundled fluid name."""
    backend, fluid_name = _split_backend(fluid)
    if backend not in _SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented in pure-JAX ChillProp")
    if any(token in fluid_name for token in ("&", "[", "]")) or fluid_name.endswith(".mix"):
        raise NotImplementedError("Mixture support is not implemented in pure-JAX ChillProp")
    key = fluid_name.lower()
    if key in _FLUID_ALIAS_OVERRIDES:
        return _FLUID_ALIAS_OVERRIDES[key]
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    return fluid_name


def _split_backend(fluid: str) -> tuple[str, str]:
    """Split a `BACKEND::Fluid` identifier into backend and fluid name."""
    if "::" in fluid:
        return tuple(fluid.split("::", 1))  # type: ignore[return-value]
    return "", fluid


def _load_fluid_data(fluid: str) -> dict[str, Any]:
    """Load a bundled fluid JSON payload by canonical name or alias."""
    key = fluid.lower()
    if key in _FLUID_DATA_CACHE:
        return _FLUID_DATA_CACHE[key]

    data_dir = resources.files("chillprop").joinpath("data")
    for candidate in data_dir.iterdir():
        if candidate.suffix.lower() != ".json":
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        fluid_data = data[0] if isinstance(data, list) else data
        info = fluid_data.get("INFO", {})
        names = {candidate.stem.lower(), str(info.get("NAME", "")).lower()}
        names.update(alias.lower() for alias in info.get("ALIASES", []))
        if key not in names:
            continue
        canonical = str(info.get("NAME", candidate.stem))
        _FLUID_DATA_CACHE[key] = fluid_data
        _FLUID_DATA_CACHE[canonical.lower()] = fluid_data
        for alias in info.get("ALIASES", []):
            _FLUID_DATA_CACHE[alias.lower()] = fluid_data
        return fluid_data

    raise FileNotFoundError(f"No bundled fluid data found for '{fluid}'")


def get_params(fluid: str) -> FluidParameters:
    """Return parsed fluid parameters, caching canonical fluid definitions."""
    canonical = _normalize_fluid_name(fluid)
    if canonical not in _FLUID_CACHE:
        data = _load_fluid_data(canonical)
        params = FluidParameters.from_json(data)
        _FLUID_CACHE[canonical] = params
        for alias in [params.name, *params.aliases]:
            _ALIAS_CACHE[alias.lower()] = params.name
    return _FLUID_CACHE[canonical]


def _is_array_like(value: Any) -> bool:
    """Return whether a value should be treated as a vectorized PropsSI input."""
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, (jax.Array, jnp.ndarray)):
        return jnp.ndim(value) > 0
    if hasattr(value, "shape"):
        return len(value.shape) > 0
    return False


def _parse_key(key: str) -> tuple[str, str | None]:
    """Normalize a CoolProp-style input key to canonical name and unit basis."""
    if "|" in key:
        _, phase = key.split("|", 1)
        raise NotImplementedError(f"Imposed phase '{phase}' is not implemented in pure-JAX ChillProp")
    if key not in _INPUT_ALIASES:
        raise ValueError(f"Input key {key} not supported")
    return _INPUT_ALIASES[key]


def _to_molar_input(params: FluidParameters, canonical: str, basis: str, value: Any) -> Any:
    """Convert mass-basis scalar inputs to the molar basis used internally."""
    if canonical == "Dmolar":
        return value if basis == "molar" else value / params.M
    if canonical in {"Hmolar", "Smolar", "Umolar"}:
        return value if basis == "molar" else value * params.M
    return value


def _gibbs_molar(params: FluidParameters, rho: Any, T: Any) -> Any:
    """Return molar Gibbs free energy."""
    return enthalpy(params, rho, T) - T * entropy(params, rho, T)


def _helmholtz_molar(params: FluidParameters, rho: Any, T: Any) -> Any:
    """Return molar Helmholtz free energy."""
    return internal_energy(params, rho, T) - T * entropy(params, rho, T)


def _compressibility_factor(params: FluidParameters, rho: Any, T: Any) -> Any:
    """Return the compressibility factor `Z`."""
    return pressure(params, rho, T) / (rho * params.R * T)


def _phase_index(params: FluidParameters, rho: Any, T: Any) -> Any:
    """Map the internal phase classifier to CoolProp-compatible phase ids."""
    P = pressure(params, rho, T)
    Tc = params.Tc
    Pc = params.Pc
    basic = get_phase(params, rho, T)
    return jnp.where(
        basic == 2,
        iphase_twophase,
        jnp.where(
            T > Tc,
            jnp.where(P > Pc, iphase_supercritical, iphase_supercritical_gas),
            jnp.where(P > Pc, iphase_supercritical_liquid, jnp.where(basic == 0, iphase_liquid, iphase_gas)),
        ),
    )


def _solve_state(params: FluidParameters, key1: str, val1: Any, key2: str, val2: Any) -> tuple[Any, Any]:
    """Solve for molar density and temperature from a supported input pair."""
    c1, b1 = _parse_key(key1)
    c2, b2 = _parse_key(key2)
    v1 = _to_molar_input(params, c1, b1, val1)
    v2 = _to_molar_input(params, c2, b2, val2)

    if {c1, c2} == {"P", "T"}:
        P = v1 if c1 == "P" else v2
        T = v2 if c1 == "P" else v1
        rho = solve_rho_PT(params, jnp.asarray(P), jnp.asarray(T))
        return rho, jnp.asarray(T)

    if {c1, c2} == {"P", "Hmolar"}:
        P = v1 if c1 == "P" else v2
        h = v2 if c1 == "P" else v1
        rho, T = solve_rhoT_Ph(params, jnp.asarray(P), jnp.asarray(h))
        return rho, T

    if {c1, c2} == {"P", "Smolar"}:
        P = v1 if c1 == "P" else v2
        s = v2 if c1 == "P" else v1
        rho, T = solve_rhoT_Ps(params, jnp.asarray(P), jnp.asarray(s))
        return rho, T

    if {c1, c2} == {"T", "Q"}:
        T = v1 if c1 == "T" else v2
        Q = v2 if c1 == "T" else v1
        rho_l, rho_v = _saturation_densities(params, jnp.asarray(T))
        rho = 1.0 / (Q / rho_v + (1.0 - Q) / rho_l)
        return rho, jnp.asarray(T)

    if {c1, c2} == {"T", "Dmolar"}:
        T = v1 if c1 == "T" else v2
        rho = v2 if c1 == "T" else v1
        return jnp.asarray(rho), jnp.asarray(T)

    raise NotImplementedError(f"Input pair ({key1}, {key2}) not yet supported in pure-JAX ChillProp")


def _saturation_densities(params: FluidParameters, T: Any) -> tuple[Any, Any]:
    """Return saturation densities, using ancillary fallbacks only where EOS VLE is known to drift."""
    ancillary_preferred = {"Methanol"}
    if (
        params.name in ancillary_preferred
        and (not params.pseudo_pure)
        and params.ancillary_rhoL is not None
        and params.ancillary_rhoV is not None
    ):
        return rhol_anc(params, T), rhov_anc(params, T)
    rho_l, rho_v = solve_vle(params, T)
    return rho_l, rho_v


def _two_phase_context(params: FluidParameters, rho: Any, T: Any) -> tuple[Any, Any, Any, Any]:
    """Return saturation properties and vapor quality for a candidate state."""
    if not isinstance(T, jax.core.Tracer):
        try:
            if np.asarray(T).shape == () and float(T) >= params.Tc:
                nan = jnp.asarray(jnp.nan)
                return nan, nan, jnp.asarray(False), jnp.asarray(jnp.nan)
        except Exception:
            pass
    T_vle = jnp.where(T < params.Tc, T, params.Tc * 0.99)
    rho_l, rho_v = _saturation_densities(params, T_vle)
    is_twophase = (T < params.Tc) & (rho >= rho_v * 0.999) & (rho <= rho_l * 1.001)
    q_calc = vapor_quality(rho, rho_l, rho_v)
    return rho_l, rho_v, is_twophase, q_calc


def _weighted_property(params: FluidParameters, rho: Any, T: Any, func, *, mass_mult: float = 1.0) -> Any:
    """Blend single-phase and two-phase property values on a mass basis if needed."""
    rho_l, rho_v, is_twophase, q_calc = _two_phase_context(params, rho, T)
    single = func(params, rho, T)
    if params.pseudo_pure:
        pl = evaluate_ancillary(params.ancillary_pL, T) if params.ancillary_pL else pressure(params, rho_l, T)
        pv = evaluate_ancillary(params.ancillary_pV, T) if params.ancillary_pV else pressure(params, rho_v, T)
        val_l = func(params, rho_l, T)
        val_v = func(params, rho_v, T)
        if func is pressure:
            two = (1.0 - q_calc) * pl + q_calc * pv
        else:
            two = (1.0 - q_calc) * val_l + q_calc * val_v
    else:
        val_l = func(params, rho_l, T)
        val_v = func(params, rho_v, T)
        blend = (1.0 - q_calc) * val_l + q_calc * val_v
        if func is pressure:
            rho_l_eq, rho_v_eq = solve_vle(params, T)
            p_eq_l = pressure(params, rho_l_eq, T)
            p_eq_v = pressure(params, rho_v_eq, T)
            two = 0.5 * (p_eq_l + p_eq_v)
        else:
            two = blend
    return jnp.where(is_twophase, two, single) * mass_mult


def _trivial_output(params: FluidParameters, out_key: str) -> float:
    """Return CoolProp-style trivial outputs that do not require a state solve."""
    trivial = {
        "Tcrit": params.Tc,
        "T_critical": params.Tc,
        "pcrit": params.Pc,
        "Pcrit": params.Pc,
        "p_critical": params.Pc,
        "rhomolar_critical": params.rhoc,
        "rhocrit": params.rhoc * params.M,
        "rhomass_critical": params.rhoc * params.M,
        "Ttriple": params.Ttriple,
        "T_triple": params.Ttriple,
        "ptriple": params.Ptriple,
        "p_triple": params.Ptriple,
        "Tmin": params.Tmin,
        "T_min": params.Tmin,
        "Tmax": params.Tmax,
        "T_max": params.Tmax,
        "Pmax": params.Pmax,
        "pmax": params.Pmax,
        "Pmin": params.Pmin,
        "P_min": params.Pmin,
        "pmin": params.Pmin,
        "acentric": params.acentric,
        "acentric_factor": params.acentric,
        "molar_mass": params.M,
        "M": params.M,
        "MolarMass": params.M,
        "gas_constant": params.R,
        "R": params.R,
        "R_u": params.R,
    }
    if out_key not in trivial:
        raise ValueError(f"Unsupported trivial output key {out_key}")
    return float(trivial[out_key])


def _evaluate_output(params: FluidParameters, out_key: str, rho: Any, T: Any) -> Any:
    """Evaluate a supported CoolProp output key at the solved state."""
    if out_key in {"D", "Density"}:
        return rho * params.M
    if out_key == "Dmolar":
        return rho
    if out_key in {"T", "Temperature"}:
        return T
    if out_key in {"P", "Pressure"}:
        return _weighted_property(params, rho, T, pressure)
    if out_key in {"H", "Enthalpy"}:
        return _weighted_property(params, rho, T, enthalpy, mass_mult=1.0 / params.M)
    if out_key == "Hmolar":
        return _weighted_property(params, rho, T, enthalpy)
    if out_key in {"S", "Entropy"}:
        return _weighted_property(params, rho, T, entropy, mass_mult=1.0 / params.M)
    if out_key == "Smolar":
        return _weighted_property(params, rho, T, entropy)
    if out_key in {"U", "InternalEnergy"}:
        return _weighted_property(params, rho, T, internal_energy, mass_mult=1.0 / params.M)
    if out_key == "Umolar":
        return _weighted_property(params, rho, T, internal_energy)
    if out_key in {"C", "CPMASS", "Cpmass"}:
        return cpmolar(params, rho, T) / params.M
    if out_key in {"Cpmolar"}:
        return cpmolar(params, rho, T)
    if out_key in {"O", "CVMASS", "Cvmass"}:
        return cvmolar(params, rho, T) / params.M
    if out_key in {"Cvmolar"}:
        return cvmolar(params, rho, T)
    if out_key in {"G", "Gmass"}:
        return _gibbs_molar(params, rho, T) / params.M
    if out_key == "Gmolar":
        return _gibbs_molar(params, rho, T)
    if out_key == "Helmholtzmass":
        return _helmholtz_molar(params, rho, T) / params.M
    if out_key == "Helmholtzmolar":
        return _helmholtz_molar(params, rho, T)
    if out_key == "Q":
        return _two_phase_context(params, rho, T)[3]
    if out_key in {"V", "viscosity"}:
        return viscosity(params, rho, T)
    if out_key in {"L", "conductivity"}:
        return thermal_conductivity(params, rho, T)
    if out_key in {"A", "speed_sound"}:
        return speed_sound(params, rho, T)
    if out_key == "Prandtl":
        cp_mass = cpmolar(params, rho, T) / params.M
        return cp_mass * viscosity(params, rho, T) / thermal_conductivity(params, rho, T)
    if out_key in {"gas_constant", "R", "R_u"}:
        return params.R
    if out_key in {"molar_mass", "M", "MolarMass"}:
        return params.M
    if out_key in {"Z", "compressibility_factor"}:
        return _compressibility_factor(params, rho, T)
    if out_key == "Phase":
        return _phase_index(params, rho, T).astype(jnp.float64)
    if out_key in _TRIVIAL_KEYS:
        return _trivial_output(params, out_key)
    derivative_match = _DERIVATIVE_RE.match(out_key)
    if derivative_match:
        raise NotImplementedError("Derivative-string PropsSI outputs are not yet implemented in pure-JAX ChillProp")
    raise ValueError(f"Output key {out_key} not supported")


def PropsSI(*args: Any) -> Any:
    """Evaluate a CoolProp-style property query for scalar or vector inputs."""
    if len(args) == 2:
        out_key, fluid = args
        params = get_params(fluid)
        return _trivial_output(params, out_key)
    if len(args) != 6:
        raise TypeError("PropsSI accepts either 2 or 6 arguments")

    out_key, key1, val1, key2, val2, fluid = args
    params = get_params(fluid)

    if _is_array_like(val1) or _is_array_like(val2):
        v1 = jnp.asarray(val1)
        v2 = jnp.asarray(val2)
        v1, v2 = jnp.broadcast_arrays(v1, v2)

        @eqx.filter_jit
        @jax.vmap
        def _batched(a, b):
            rho, T = _solve_state(params, key1, a, key2, b)
            return _evaluate_output(params, out_key, rho, T)

        return _batched(v1, v2)

    rho, T = _solve_state(params, key1, val1, key2, val2)
    return _evaluate_output(params, out_key, rho, T)


def PhaseSI(name1: str, prop1: float, name2: str, prop2: float, fluid: str) -> str:
    """Return the string phase label for a CoolProp-style state specification."""
    phase = int(PropsSI("Phase", name1, prop1, name2, prop2, fluid))
    reverse = {
        iphase_liquid: "liquid",
        iphase_supercritical: "supercritical",
        iphase_supercritical_gas: "supercritical_gas",
        iphase_supercritical_liquid: "supercritical_liquid",
        iphase_critical_point: "critical_point",
        iphase_gas: "gas",
        iphase_twophase: "twophase",
        iphase_unknown: "unknown",
        iphase_not_imposed: "not_imposed",
    }
    return reverse.get(phase, "unknown")


def Props1SI(fluid: str, output: str) -> float:
    """Return a state-independent CoolProp-style property."""
    return PropsSI(output, fluid)


def get_phase_index(key: str) -> int:
    """Return the integer code for a CoolProp phase identifier string."""
    if key not in _PHASE_INDEX:
        raise ValueError(f"Invalid phase key {key}")
    return _PHASE_INDEX[key]


def set_reference_state(*_args: Any, **_kwargs: Any) -> None:
    """Reject mutable reference-state changes for the pure-JAX backend."""
    raise NotImplementedError("Reference-state mutation is not yet implemented in pure-JAX ChillProp")


class AbstractState:
    """Pure-JAX HEOS-like state wrapper for supported pure and pseudo-pure fluids."""

    def __init__(self, backend: str, fluid: str):
        """Initialize an abstract state for a supported backend and fluid."""
        if backend not in _SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented in pure-JAX ChillProp")
        self.params = get_params(fluid)
        self.rho = None
        self.temperature = None

    def update(self, input_pair: int, val1: float, val2: float):
        """Update the state from a supported CoolProp input-pair constant."""
        if input_pair == PT_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "P", val1, "T", val2)
        elif input_pair == HmassP_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "H", val1, "P", val2)
        elif input_pair == HmolarP_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "Hmolar", val1, "P", val2)
        elif input_pair == PSmass_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "P", val1, "S", val2)
        elif input_pair == PSmolar_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "P", val1, "Smolar", val2)
        elif input_pair == QT_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "Q", val1, "T", val2)
        elif input_pair == DmassT_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "D", val1, "T", val2)
        elif input_pair == DmolarT_INPUTS:
            self.rho, self.temperature = _solve_state(self.params, "Dmolar", val1, "T", val2)
        else:
            raise NotImplementedError(f"AbstractState.update for pair {input_pair} not implemented in pure-JAX ChillProp")

    def _require_state(self):
        """Ensure `update` has been called before reading state-dependent outputs."""
        if self.rho is None or self.temperature is None:
            raise ValueError("State has not been updated")

    def rhomolar(self):
        """Return molar density."""
        self._require_state()
        return self.rho

    def rhomass(self):
        """Return mass density."""
        self._require_state()
        return self.rho * self.params.M

    def hmolar(self):
        """Return molar enthalpy."""
        self._require_state()
        return enthalpy(self.params, self.rho, self.temperature)

    def hmass(self):
        """Return mass-specific enthalpy."""
        return self.hmolar() / self.params.M

    def smolar(self):
        """Return molar entropy."""
        self._require_state()
        return entropy(self.params, self.rho, self.temperature)

    def smass(self):
        """Return mass-specific entropy."""
        return self.smolar() / self.params.M

    def umolar(self):
        """Return molar internal energy."""
        self._require_state()
        return internal_energy(self.params, self.rho, self.temperature)

    def umass(self):
        """Return mass-specific internal energy."""
        return self.umolar() / self.params.M

    def p(self):
        """Return pressure."""
        self._require_state()
        return pressure(self.params, self.rho, self.temperature)

    def T(self):
        """Return temperature."""
        self._require_state()
        return self.temperature

    def Q(self):
        """Return vapor quality."""
        self._require_state()
        return _evaluate_output(self.params, "Q", self.rho, self.temperature)

    def cpmolar(self):
        """Return molar constant-pressure heat capacity."""
        self._require_state()
        return cpmolar(self.params, self.rho, self.temperature)

    def cpmass(self):
        """Return mass-specific constant-pressure heat capacity."""
        return self.cpmolar() / self.params.M

    def cvmolar(self):
        """Return molar constant-volume heat capacity."""
        self._require_state()
        return cvmolar(self.params, self.rho, self.temperature)

    def cvmass(self):
        """Return mass-specific constant-volume heat capacity."""
        return self.cvmolar() / self.params.M

    def viscosity(self):
        """Return dynamic viscosity."""
        self._require_state()
        return viscosity(self.params, self.rho, self.temperature)

    def conductivity(self):
        """Return thermal conductivity."""
        self._require_state()
        return thermal_conductivity(self.params, self.rho, self.temperature)

    def keyed_output(self, key: int):
        """Return a CoolProp keyed output for the current state."""
        self._require_state()
        mapping = {
            iDmolar: self.rhomolar(),
            iDmass: self.rhomass(),
            iT: self.T(),
            iP: self.p(),
            iHmolar: self.hmolar(),
            iHmass: self.hmass(),
            iSmolar: self.smolar(),
            iSmass: self.smass(),
            iUmolar: self.umolar(),
            iUmass: self.umass(),
            iCpmolar: self.cpmolar(),
            iCpmass: self.cpmass(),
            iCvmolar: self.cvmolar(),
            iCvmass: self.cvmass(),
            iQ: self.Q(),
            iviscosity: self.viscosity(),
            iconductivity: self.conductivity(),
            ispeed_sound: speed_sound(self.params, self.rho, self.temperature),
            iPrandtl: self.cpmass() * self.viscosity() / self.conductivity(),
            iZ: _compressibility_factor(self.params, self.rho, self.temperature),
            iPhase: _phase_index(self.params, self.rho, self.temperature).astype(jnp.float64),
            imolar_mass: self.params.M,
            igas_constant: self.params.R,
            iacentric_factor: self.params.acentric,
            irhomolar_critical: self.params.rhoc,
            irhomass_critical: self.params.rhoc * self.params.M,
            iT_critical: self.params.Tc,
            iP_critical: self.params.Pc,
            iT_triple: self.params.Ttriple,
            iP_triple: self.params.Ptriple,
            iT_min: self.params.Tmin,
            iT_max: self.params.Tmax,
            iP_min: self.params.Pmin,
            iP_max: self.params.Pmax,
        }
        if key not in mapping:
            raise ValueError(f"Key {key} not supported in pure-JAX ChillProp")
        return mapping[key]
