import numpy as np
import pytest
import jax
import jax.numpy as jnp

import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from fluid_catalog import SUPPORTED_FLUIDS, TRANSPORT_VALIDATED_FLUIDS, TWOPHASE_VALIDATED_FLUIDS

TRIVIAL_KEYS = [
    "Tcrit",
    "pcrit",
    "rhomolar_critical",
    "Ttriple",
    "ptriple",
    "Tmin",
    "Tmax",
    "Pmax",
    "acentric",
    "molar_mass",
    "gas_constant",
]

CORE_SINGLE_PHASE_OUTPUTS = [
    "D",
    "Dmolar",
    "P",
    "H",
    "Hmolar",
    "S",
    "Smolar",
    "U",
    "Umolar",
    "C",
    "Cpmolar",
    "O",
    "Cvmolar",
    "G",
    "Gmolar",
    "Helmholtzmass",
    "Helmholtzmolar",
    "A",
    "Z",
    "Phase",
]

TRANSPORT_OUTPUTS = [
    "V",
    "L",
    "Prandtl",
]

TWOPHASE_OUTPUTS = [
    "D",
    "Dmolar",
    "P",
    "H",
    "Hmolar",
    "S",
    "Smolar",
    "U",
    "Umolar",
    "Q",
]

STRICT_RTOL = 1e-9
STRICT_ATOL = 1e-12
OUTPUT_RTOL = {
    "A": 1e-7,
    "C": 5e-7,
    "Cpmolar": 5e-7,
    "V": 5e-3,
    "L": 5e-3,
    "Prandtl": 5e-3,
    "Q": 1e-4,
    "Z": 1e-7,
}

TRIVIAL_RTOL = 3e-2
CORE_RTOL = 5e-8
TWOPHASE_RTOL = 1e-3


def _coolprop_vectorized(output, key1, values1, key2, values2, fluid):
    values1 = np.asarray(values1, dtype=float)
    values2 = np.asarray(values2, dtype=float)
    values1, values2 = np.broadcast_arrays(values1, values2)
    out = np.empty(values1.shape, dtype=float)
    for idx in np.ndindex(values1.shape):
        out[idx] = CP.PropsSI(output, key1, float(values1[idx]), key2, float(values2[idx]), fluid)
    return out


def _assert_close(actual, expected, *, rtol=STRICT_RTOL, atol=STRICT_ATOL):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"max_abs={np.max(np.abs(actual - expected)):.3e}, "
        f"max_rel={np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), atol)):.3e}"
    )


def _coolprop_trivial(key, fluid):
    if key == "Pmax":
        state = CP.AbstractState("HEOS", fluid)
        return float(state.keyed_output(CP.iP_max))
    if key == "Pmin":
        state = CP.AbstractState("HEOS", fluid)
        return float(state.keyed_output(CP.iP_min))
    return float(CP.PropsSI(key, fluid))


def _chillprop_batched(output, key1, values1, key2, values2, fluid):
    values1 = np.asarray(values1, dtype=float)
    values2 = np.asarray(values2, dtype=float)
    values1, values2 = np.broadcast_arrays(values1, values2)
    out = np.empty(values1.shape, dtype=float)
    with jax.disable_jit():
        for idx in np.ndindex(values1.shape):
            out[idx] = CH.PropsSI(output, key1, float(values1[idx]), key2, float(values2[idx]), fluid)
    return out


def _single_phase_grid(fluid):
    Tc = CP.PropsSI("Tcrit", fluid)
    Pc = CP.PropsSI("pcrit", fluid)
    Tmax = CP.PropsSI("Tmax", fluid)
    Tmin = CP.PropsSI("Tmin", fluid)

    T_low = min(max(Tc * 1.05, Tmin + 10.0), Tmax * 0.65)
    T_high = min(max(Tc * 1.35, Tmin + 20.0), Tmax * 0.8)
    T_candidates = np.array([T_low, T_high], dtype=float)
    P_candidates = np.array([min(1.1 * Pc, 2.5e7), min(2.0 * Pc, 5e7)], dtype=float)
    TT, PP = np.meshgrid(T_candidates, P_candidates, indexing="ij")
    return TT.reshape(-1), PP.reshape(-1)


def _twophase_grid(fluid):
    Ttriple = CP.PropsSI("Ttriple", fluid)
    Tc = CP.PropsSI("Tcrit", fluid)
    T_vals = np.array(
        [
            Ttriple + 0.25 * (Tc - Ttriple),
            Ttriple + 0.65 * (Tc - Ttriple),
        ],
        dtype=float,
    )
    Q_vals = np.array([0.0, 0.5, 1.0], dtype=float)
    TT, QQ = np.meshgrid(T_vals, Q_vals, indexing="ij")
    return TT.reshape(-1), QQ.reshape(-1)


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_trivial_outputs_all_supported_materials(fluid):
    for key in TRIVIAL_KEYS:
        _assert_close(CH.PropsSI(key, fluid), _coolprop_trivial(key, fluid), rtol=TRIVIAL_RTOL, atol=1e-12)


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_single_phase_grid_parity_all_supported_core_features(fluid):
    T, P = _single_phase_grid(fluid)
    params = CH.get_params(fluid)
    solve_pt = jax.jit(lambda t, p: CH._solve_state(params, "T", t, "P", p))
    eval_outputs = jax.jit(
        lambda rho, temp: jnp.stack([CH._evaluate_output(params, output, rho, temp) for output in CORE_SINGLE_PHASE_OUTPUTS])
    )
    for t, p in zip(T, P):
        rho, temp = solve_pt(float(t), float(p))
        chill_values = np.asarray(eval_outputs(rho, temp), dtype=float)
        for output, chill in zip(CORE_SINGLE_PHASE_OUTPUTS, chill_values):
            ref = float(CP.PropsSI(output, "T", float(t), "P", float(p), fluid))
            _assert_close(chill, ref, rtol=OUTPUT_RTOL.get(output, CORE_RTOL), atol=STRICT_ATOL)


@pytest.mark.parametrize("fluid", TRANSPORT_VALIDATED_FLUIDS)
def test_single_phase_grid_parity_transport_features(fluid):
    T, P = _single_phase_grid(fluid)
    params = CH.get_params(fluid)
    solve_pt = jax.jit(lambda t, p: CH._solve_state(params, "T", t, "P", p))
    eval_outputs = jax.jit(
        lambda rho, temp: jnp.stack([CH._evaluate_output(params, output, rho, temp) for output in TRANSPORT_OUTPUTS])
    )
    for t, p in zip(T, P):
        rho, temp = solve_pt(float(t), float(p))
        chill_values = np.asarray(eval_outputs(rho, temp), dtype=float)
        for output, chill in zip(TRANSPORT_OUTPUTS, chill_values):
            ref = float(CP.PropsSI(output, "T", float(t), "P", float(p), fluid))
            _assert_close(chill, ref, rtol=OUTPUT_RTOL[output], atol=STRICT_ATOL)


@pytest.mark.parametrize("fluid", TWOPHASE_VALIDATED_FLUIDS)
def test_two_phase_grid_parity_all_supported_features(fluid):
    T, Q = _twophase_grid(fluid)
    params = CH.get_params(fluid)
    solve_tq = jax.jit(lambda t, q: CH._solve_state(params, "T", t, "Q", q))
    eval_outputs = jax.jit(lambda rho, temp: jnp.stack([CH._evaluate_output(params, output, rho, temp) for output in TWOPHASE_OUTPUTS]))
    for t, q in zip(T, Q):
        rho, temp = solve_tq(float(t), float(q))
        chill_values = np.asarray(eval_outputs(rho, temp), dtype=float)
        for output, chill in zip(TWOPHASE_OUTPUTS, chill_values):
            ref = float(CP.PropsSI(output, "T", float(t), "Q", float(q), fluid))
            _assert_close(chill, ref, rtol=OUTPUT_RTOL.get(output, TWOPHASE_RTOL), atol=STRICT_ATOL)


def test_supported_keyed_outputs():
    T, P = _single_phase_grid("Nitrogen")
    state = CH.AbstractState("HEOS", "Nitrogen")
    state.update(CH.PT_INPUTS, float(P[0]), float(T[0]))
    for key in [
        CH.iT,
        CH.iP,
        CH.iDmolar,
        CH.iDmass,
        CH.iHmolar,
        CH.iHmass,
        CH.iSmolar,
        CH.iSmass,
        CH.iUmolar,
        CH.iUmass,
        CH.iCpmolar,
        CH.iCpmass,
        CH.iCvmolar,
        CH.iCvmass,
        CH.iviscosity,
        CH.iconductivity,
        CH.ispeed_sound,
        CH.iPrandtl,
        CH.iZ,
        CH.iPhase,
        CH.imolar_mass,
        CH.igas_constant,
        CH.iacentric_factor,
        CH.irhomolar_critical,
        CH.irhomass_critical,
        CH.iT_critical,
        CH.iP_critical,
        CH.iT_triple,
        CH.iP_triple,
        CH.iT_min,
        CH.iT_max,
        CH.iP_min,
        CH.iP_max,
    ]:
        value = state.keyed_output(key)
        assert np.isfinite(float(value))


def test_vectorized_propssi_smoke():
    T = np.array([280.0, 300.0, 320.0])
    P = np.array([5e5, 1e6, 2e6])
    rho = CH.PropsSI("D", "T", T, "P", P, "Nitrogen")
    ref = _coolprop_vectorized("D", "T", T, "P", P, "Nitrogen")
    _assert_close(rho, ref)


def test_unsupported_backend_is_explicit():
    with pytest.raises(NotImplementedError):
        CH.PropsSI("D", "T", 300.0, "P", 101325.0, "REFPROP::Water")


def test_unsupported_mixture_is_explicit():
    with pytest.raises(NotImplementedError):
        CH.PropsSI("D", "T", 300.0, "P", 101325.0, "HEOS::Propane[0.5]&Ethane[0.5]")


def test_unsupported_phase_hint_is_explicit():
    with pytest.raises(NotImplementedError):
        CH.PropsSI("D", "T|liquid", 300.0, "P", 101325.0, "Water")


def test_reference_state_mutation_is_explicit():
    with pytest.raises(NotImplementedError):
        CH.set_reference_state("Nitrogen", "ASHRAE")
