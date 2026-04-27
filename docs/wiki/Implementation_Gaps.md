# Implementation Gaps Relative to CoolProp

This page describes the deliberate and current functional differences between ChillProp and CoolProp.

## Backend Coverage

Supported:

- Default backend
- `HEOS`

Not implemented:

- `REFPROP`
- Incompressible backends
- Tabular backends
- Cubic EOS backends
- Any other backend-specific extensions exposed by CoolProp

Calls that specify unsupported backends raise `NotImplementedError`.

## Fluid and Mixture Coverage

Supported:

- Bundled pure and pseudo-pure fluids shipped in `src/chillprop/data`

Not implemented:

- Mixtures such as `HEOS::Propane[0.5]&Ethane[0.5]`
- Dynamic discovery of the full CoolProp fluid database at runtime

The package currently behaves as a curated subset rather than a drop-in replacement for the full CoolProp material catalog.

## Input-Pair Coverage

Implemented high-level state specifications:

- `P`, `T`
- `P`, `H`
- `P`, `S`
- `T`, `Q`
- `T`, `D`
- `T`, `Dmolar`

Not implemented:

- The broader CoolProp matrix of input pairs such as `PU`, `HS`, `DQ`, and many others
- Phase-imposed input keys such as `T|liquid`

Unsupported input pairs raise `NotImplementedError`.

## Output Coverage

Implemented output groups:

- Trivial fluid constants such as critical, triple-point, and molar-mass properties
- Core thermodynamic properties
- Phase index and phase string helpers
- Transport outputs required by the current test matrix

Not implemented:

- Derivative-string outputs such as `d(Hmass)/d(P)|T`
- The full keyed-output catalog exposed by CoolProp
- Every convenience alias accepted by CoolProp

The authoritative mapping lives in [src/chillprop/highlevel.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/highlevel.py).

## `AbstractState` Coverage

Implemented:

- Construction with supported backends and fluids
- `update()` for the input-pair subset listed above
- Common scalar property accessors
- A subset of keyed outputs

Not implemented:

- The full `AbstractState` mutation and inspection surface available in CoolProp
- General phase-envelope tools
- Comprehensive mixture state operations

## Reference-State Mutation

CoolProp exposes mutable reference-state APIs. ChillProp currently does not.

- `set_reference_state(...)` raises `NotImplementedError`
- Reference-state behavior is fixed by the bundled fluid definitions

## Transport Coverage and Validation Scope

Transport-model implementations exist for more fluids than the minimum automated transport test grid. The documented automated transport parity grid currently covers:

- `Argon`
- `Hydrogen`
- `n-Decane`
- `Nitrogen`
- `Oxygen`
- `Propane`

Two-phase property support is also broader than the strict automated saturation-parity subset documented in [Validation](Validation.md). This distinction matters for contributors because "implemented" and "covered by the regression suite" are not identical statements.

## Numerical Scope

ChillProp is built around JAX transforms and differentiable Newton solves. That brings some current constraints:

- Solver behavior is tuned for the documented input-pair set rather than the full CoolProp state-specification matrix
- Unsupported features are surfaced explicitly instead of silently approximated
- Validation is strongest on the documented supported set and should be expanded alongside feature additions

## Contributor Guidance

When closing a compatibility gap:

1. Update the public API implementation.
2. Add or extend regression coverage in `tests`.
3. Update [Validation](Validation.md) if the change affects documented parity scope.
4. Update this page so the compatibility boundary remains explicit.
