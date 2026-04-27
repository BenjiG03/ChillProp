import jax
import numpy as np
import pytest

import chillprop.highlevel as CH
from fluid_catalog import SUPPORTED_FLUIDS


jax.config.update("jax_enable_x64", True)

GRAD_RTOL = 1e-6
FD_T_STEP = 1e-4
FD_P_STEP = 1.0
FD_RHO_STEP_FACTOR = 1e-6


def _rel_err(a, b):
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


def _stable_temperature(fluid, scale, margin):
    Tmin = CH.PropsSI("Tmin", fluid)
    Tmax = CH.PropsSI("Tmax", fluid)
    Tc = CH.PropsSI("Tcrit", fluid)
    return min(max(Tc * scale, Tmin + margin), Tmax * 0.8)


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_density_gradient_vs_finite_difference(fluid):
    T0 = _stable_temperature(fluid, 1.35, 10.0)
    P0 = max(2e5, 0.35 * CH.PropsSI("pcrit", fluid))

    def rho_of_T(T):
        return CH.PropsSI("D", "T", T, "P", P0, fluid)

    grad_val = float(jax.grad(rho_of_T)(T0))
    fd_val = float((rho_of_T(T0 + FD_T_STEP) - rho_of_T(T0 - FD_T_STEP)) / (2.0 * FD_T_STEP))
    assert _rel_err(grad_val, fd_val) < GRAD_RTOL, f"{fluid}: grad={grad_val}, fd={fd_val}"


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_enthalpy_gradient_vs_finite_difference(fluid):
    T0 = _stable_temperature(fluid, 1.25, 15.0)
    P0 = max(2e5, 0.4 * CH.PropsSI("pcrit", fluid))

    def h_of_P(P):
        return CH.PropsSI("H", "T", T0, "P", P, fluid)

    grad_val = float(jax.grad(h_of_P)(P0))
    fd_val = float((h_of_P(P0 + FD_P_STEP) - h_of_P(P0 - FD_P_STEP)) / (2.0 * FD_P_STEP))
    assert _rel_err(grad_val, fd_val) < GRAD_RTOL, f"{fluid}: grad={grad_val}, fd={fd_val}"


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_pressure_gradient_vs_finite_difference(fluid):
    T0 = _stable_temperature(fluid, 1.2, 10.0)
    P0 = max(2e5, 0.45 * CH.PropsSI("pcrit", fluid))
    rho0 = CH.PropsSI("Dmolar", "T", T0, "P", P0, fluid)
    drho = max(abs(rho0) * FD_RHO_STEP_FACTOR, 1e-3)

    def p_of_rho(rho):
        return CH.PropsSI("P", "T", T0, "Dmolar", rho, fluid)

    grad_val = float(jax.grad(p_of_rho)(rho0))
    fd_val = float((p_of_rho(rho0 + drho) - p_of_rho(rho0 - drho)) / (2.0 * drho))
    assert _rel_err(grad_val, fd_val) < GRAD_RTOL, f"{fluid}: grad={grad_val}, fd={fd_val}"
