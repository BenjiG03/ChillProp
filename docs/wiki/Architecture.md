# Architecture

ChillProp is structured as a Helmholtz-energy property engine with a thin CoolProp-like compatibility layer. The implementation is centered on typed fluid metadata, functional state evaluation, and JAX transformations.

## Design Goals

- Preserve a familiar subset of the CoolProp high-level API
- Keep core thermodynamic kernels differentiable
- Separate fluid metadata, equation-of-state kernels, state solvers, and API translation
- Bundle the runtime fluid database with the package instead of depending on an external CoolProp checkout

## Module Responsibilities

### `parameters.py`

`parameters.py` defines the typed data model used throughout the package.

- `FluidParameters` stores EOS constants, ancillary correlations, and transport metadata
- Ideal-gas and residual Helmholtz terms are represented as Equinox modules
- Viscosity and conductivity models are parsed into typed transport structures
- `FluidParameters.from_json()` converts bundled CoolProp-style JSON into runtime objects

This module is the bridge between serialized fluid data and executable thermodynamic kernels.

### `heos.py`

`heos.py` evaluates the ideal-gas and residual Helmholtz energy contributions.

- `alpha0_*` functions evaluate ideal-gas term families
- `alphar_*` functions evaluate residual term families
- `alpha0()` and `alphar()` sum all active terms for a fluid
- `evaluate_alpha0()` and `evaluate_alphar()` expose `(rho, T)` wrappers around the reduced variables

The functions in this module are intentionally small because higher-level properties are obtained by differentiation rather than by duplicating algebra in multiple places.

### `core.py`

`core.py` converts Helmholtz energy and its derivatives into thermodynamic properties.

- `get_alpha_and_derivs()` computes first derivatives of `alpha0` and `alphar`
- `pressure()`, `enthalpy()`, `entropy()`, and `internal_energy()` implement the standard Helmholtz relations
- `cvmolar()`, `cpmolar()`, and `speed_sound()` derive second-order properties with JAX automatic differentiation
- `props()` provides a compact bundle used by a few tests and diagnostics

The central architectural choice is that property formulas stay close to the textbook reduced-variable relations, while JAX supplies the derivative machinery.

### `phases.py`

`phases.py` contains ancillary evaluation and phase-equilibrium helpers.

- `evaluate_ancillary()` evaluates CoolProp-style ancillary correlations
- `rhol_anc()`, `rhov_anc()`, and `psat_anc()` expose common saturation correlations
- `solve_vle()` performs a two-variable Newton solve for liquid and vapor saturation densities
- `get_phase()` maps a state to a coarse phase classification

The two-phase logic is intentionally separate from the single-phase solver because most high-level property calls only need a saturation context when the state falls inside the coexistence region.

### `solver.py`

`solver.py` implements differentiable state inversion.

- `find_rho_PT()` solves `P(rho, T) = P_target` with Newton iteration and a custom VJP
- `solve_rho_PT()` builds phase-aware initial guesses from ancillaries and ideal-gas behavior
- `solve_2d()` provides a generic Newton loop for coupled unknowns
- `solve_rhoT_Ph()` and `solve_rhoT_Ps()` invert `(P, h)` and `(P, s)`

The custom VJP on the `PT` density solve avoids backpropagating through every Newton iteration while still exposing useful sensitivities for downstream optimization.

### `transport.py`

`transport.py` implements transport models parsed from the bundled JSON data.

- Dilute viscosity models include reduced-temperature power series and collision-integral forms
- Higher-order viscosity support includes friction-theory and modified Batschinski-Hildebrand branches
- Conductivity support includes dilute, residual, and critical-enhancement terms where available
- Transport functions consume the same `FluidParameters` object as the EOS kernels

Transport coverage is broader than the minimum public test matrix, but the validation scope is explicitly narrower than thermodynamic property coverage.

### `highlevel.py`

`highlevel.py` is the public API adapter.

- Loads bundled fluid JSON via `importlib.resources`
- Caches parsed `FluidParameters`
- Normalizes CoolProp-style fluid names and aliases
- Solves state variables from supported input pairs
- Maps output keys to core thermodynamic, phase, or transport functions
- Implements `PropsSI`, `Props1SI`, `PhaseSI`, constants, and the supported subset of `AbstractState`

This module is the best starting point when extending user-facing behavior because it expresses the package's current compatibility contract.

## Data Flow

For a typical `PropsSI(output, key1, val1, key2, val2, fluid)` call:

1. `highlevel.get_params()` loads and caches the fluid definition.
2. `highlevel._solve_state()` selects the state inversion path for the input pair.
3. `solver.solve_rho_PT()` or another solver returns `(rho, T)` in molar units.
4. `highlevel._evaluate_output()` dispatches to `core.py`, `phases.py`, or `transport.py`.
5. If the output may enter the coexistence region, `_weighted_property()` blends saturated liquid and vapor values when appropriate.

## Units and Conventions

- Internal density is molar density in `mol/m^3`
- Internal energy-like properties are primarily molar quantities in `J/mol`
- Mass-basis compatibility is provided at the public API boundary
- Reduced variables use `tau = Tr / T` and `delta = rho / rhor`

Contributors adding new outputs should check whether the result belongs naturally on a molar or mass basis before exposing aliases through `highlevel.py`.

## Adding a Fluid

1. Add a CoolProp-style JSON file to `src/chillprop/data`.
2. Confirm `FluidParameters.from_json()` can parse all relevant EOS and transport terms.
3. Add or extend automated parity coverage in `tests`.
4. Regenerate validation figures if the fluid is part of the documented supported set.

## Extending the API

Typical extension points:

- Add new input aliases in `_INPUT_ALIASES`
- Extend `_solve_state()` for new input-pair combinations
- Extend `_evaluate_output()` for new output keys
- Add additional `AbstractState.update()` branches and keyed outputs where parity is intended

The compatibility ceiling is documented in [Implementation Gaps](Implementation_Gaps.md) and should be updated whenever the public API surface changes.
