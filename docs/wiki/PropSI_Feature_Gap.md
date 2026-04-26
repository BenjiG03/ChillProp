# `ChillProp.PropSI` vs `CoolProp.PropsSI`

This document compares the current `ChillProp` high-level API to CoolProp's documented `PropsSI` contract and identifies what must be added for `ChillProp.PropSI` to be feature-complete.

## Scope

This is specifically about `PropsSI`, not full library parity.

- `ChillProp` implementation reviewed: [chillprop/highlevel.py](/c:/Users/Benji/Documents/ChillProp/chillprop/highlevel.py:33)
- CoolProp high-level API reference: [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:24)
- CoolProp parameter and input-pair enums: [CoolProp/include/DataStructures.h](/c:/Users/Benji/Documents/ChillProp/CoolProp/include/DataStructures.h:64)

## Current `ChillProp.PropSI` surface

The current implementation supports only these input pairs:

- `P,T`
- `P,Hmass` and `P,Hmolar`
- `P,Smass` and `P,Smolar`
- `T,Q`
- `T,Dmass` and `T,Dmolar`

That dispatch is hard-coded in [chillprop/highlevel.py](/c:/Users/Benji/Documents/ChillProp/chillprop/highlevel.py:69).

The current output set is limited to:

- `D`, `Dmolar`
- `T`, `P`
- `H`, `Hmolar`
- `S`, `Smolar`
- `U`, `Umolar`
- `Q`
- `V` viscosity
- `L` conductivity
- `A` speed of sound
- `R`
- `M`

That selection is hard-coded in [chillprop/highlevel.py](/c:/Users/Benji/Documents/ChillProp/chillprop/highlevel.py:132).

## Observed missing behavior

These calls work in CoolProp but currently fail in `ChillProp`:

```python
CHP('D', 'P', 101325, 'Q', 0, 'Water')          # P,Q input pair
CHP('H', 'P', 101325, 'D', 997, 'Water')        # P,D input pair
CHP('Tcrit', '', 0, '', 0, 'Water')             # trivial parameter
CHP('C', 'P', 101325, 'T', 300, 'Water')        # cp output
CHP('Phase', 'P', 101325, 'Q', 0, 'Water')      # phase output
CHP('d(Hmass)/d(T)|P', 'P', 101325, 'T', 300, 'Water')
CHP('D', 'T|liquid', 300, 'P', 101325, 'Water') # imposed phase
```

Observed errors from the current implementation:

- `NotImplementedError: Input pair (P, Q) not yet supported`
- `NotImplementedError: Input pair (P, rho) not yet supported`
- `NotImplementedError: Input pair (None, None) not yet supported`
- `ValueError: Output key C not supported`
- `ValueError: Output key d(Hmass)/d(T)|P not supported`
- `NotImplementedError: Input pair (None, P) not yet supported`

## What CoolProp `PropsSI` supports that `ChillProp` does not

### 1. The full input-pair matrix

CoolProp exposes these `update()` input pairs in [CoolProp/include/DataStructures.h](/c:/Users/Benji/Documents/ChillProp/CoolProp/include/DataStructures.h:280):

- `Q,T`
- `P,Q`
- `Q,Smolar`
- `Q,Smass`
- `Hmolar,Q`
- `Hmass,Q`
- `Dmolar,Q`
- `Dmass,Q`
- `P,T`
- `Dmolar,T`
- `Dmass,T`
- `Hmolar,T`
- `Hmass,T`
- `Smolar,T`
- `Smass,T`
- `T,Umolar`
- `T,Umass`
- `Dmolar,P`
- `Dmass,P`
- `Hmolar,P`
- `Hmass,P`
- `P,Smolar`
- `P,Smass`
- `P,Umolar`
- `P,Umass`
- `Hmolar,Smolar`
- `Hmass,Smass`
- `Smolar,Umolar`
- `Smass,Umass`
- `Dmolar,Hmolar`
- `Dmass,Hmass`
- `Dmolar,Smolar`
- `Dmass,Smass`
- `Dmolar,Umolar`
- `Dmass,Umass`

`ChillProp` currently implements 5 structural cases out of that full set.

What must be added:

- A general input-key parser that distinguishes mass and molar forms instead of collapsing both onto `'rho'`, `'h'`, `'s'`, and `'u'`.
- Flash solvers for all missing state pairs.
- Saturation-path solvers for `P,Q`, `H,Q`, `S,Q`, and `D,Q`.
- Single-phase solvers for `P,D`, `P,U`, `T,U`, `H,S`, `D,H`, `D,S`, and `D,U`.

### 2. Trivial outputs and the overloaded two-argument form

CoolProp documents trivial-property support and the Python two-argument overload in [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:138).

Examples:

- `PropsSI("Tcrit", "Water")`
- `PropsSI("pcrit", "Water")`
- `PropsSI("Tcrit", "", 0, "", 0, "Water")`

What must be added:

- Support for the 2-argument Python signature.
- Detection of trivial outputs before attempting a flash calculation.
- A lookup table for fluid constants and metadata-backed scalars such as:
  - `Tcrit`, `pcrit`, `rhomolar_critical`, `rhocrit`
  - `Ttriple`, `ptriple`
  - `Tmin`, `Tmax`, `Pmin`, `Pmax`
  - `acentric`
  - reducing-state properties
  - any other trivial keys expected from CoolProp's parameter table

### 3. Much broader output/property coverage

CoolProp's parameter enum includes many outputs beyond what `ChillProp` currently returns; see [CoolProp/include/DataStructures.h](/c:/Users/Benji/Documents/ChillProp/CoolProp/include/DataStructures.h:64).

Major missing output categories:

- Heat capacities: `C`, `Cpmass`, `Cpmolar`, `C0`, `Cp0mass`, `Cp0molar`, `O`, `Cvmass`, `Cvmolar`
- Free energies: `G`, `Gmass`, `Gmolar`, `Helmholtzmass`, `Helmholtzmolar`
- Residual and ideal-gas splits: `Hmolar_residual`, `Smolar_residual`, `Gmolar_residual`, `Hmass_idealgas`, `Smass_idealgas`, `Umass_idealgas`, and molar equivalents
- Transport-derived outputs: `Prandtl`, `surface_tension`
- Response functions: `isothermal_compressibility`, `isobaric_expansion_coefficient`, `isentropic_expansion_coefficient`
- EOS outputs: `Z`, `Bvirial`, `Cvirial`, `dBvirial_dT`, `dCvirial_dT`, `PIP`
- Phase output: `Phase`
- Environmental and fluid metadata outputs where CoolProp exposes them through `PropsSI`

What must be added:

- The underlying thermodynamic functions for all missing scalar properties.
- A richer alias map matching CoolProp names, abbreviations, and mass/molar distinctions.
- Output validation and unit-consistent mass/molar conversions.

### 4. Phase-imposed input syntax

CoolProp allows imposed phase strings like `T|liquid` and `P|gas`; see [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:73).

What must be added:

- Parsing of `key|phase` input strings.
- Validation that exactly one input key carries a phase hint.
- Mapping of CoolProp phase aliases:
  - `liquid`
  - `gas`
  - `twophase`
  - `supercritical_liquid`
  - `supercritical_gas`
  - `supercritical`
  - `not_imposed`
  - `phase_*` and `iphase_*` aliases
- Propagation of phase hints into the flash routines and initial-guess logic.

### 5. Generalized derivative-string outputs

CoolProp supports derivative strings in `PropsSI`; see [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:269).

Missing classes of derivative output:

- First single-phase derivatives like `d(Hmass)/d(T)|P`
- Second single-phase derivatives like `d(d(Hmass)/d(T)|P)/d(Hmass)|P`
- Saturation derivatives like `d(Hmolar)/d(T)|sigma`

What must be added:

- A parser for CoolProp derivative-string grammar.
- First-derivative kernels over the `AbstractState` property set.
- Second-derivative kernels.
- Saturation-curve derivative support for `Q in {0, 1}`.
- Error handling that rejects invalid two-phase derivative requests where CoolProp also documents restrictions.

### 6. Phase queries through `PropsSI`

CoolProp supports `PropsSI("Phase", ...)` and a separate `PhaseSI(...)`; see [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:164).

`ChillProp` imports `get_phase` but does not expose phase through `PropsSI`.

What must be added:

- `PropsSI("Phase", ...)` returning the same numeric phase index style as CoolProp.
- A `PhaseSI(...)` helper returning the phase string.
- Phase-index constants or a `get_phase_index()` compatibility helper.

### 7. Backend prefixes and mixture syntax

CoolProp `PropsSI` accepts backend-prefixed fluids and mixture strings such as:

- `HEOS::R32[0.697615]&R125[0.302385]`
- `REFPROP::...`
- `IF97::Water`
- predefined mixtures like `Air.mix`

These are part of the documented high-level API in [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:347).

What must be added:

- Fluid-string parsing for `BACKEND::fluid`.
- Mixture-string parsing for `component[fraction]&component[fraction]`.
- Composition handling in the state model.
- Backend dispatch, or explicit documented non-support if `ChillProp` only intends to emulate `HEOS`.

Without this, `PropsSI` cannot be considered feature-complete with respect to CoolProp.

### 8. Reference-state compatibility implications

CoolProp's `PropsSI` behavior is affected by `set_reference_state(...)`; see [CoolProp/Web/coolprop/HighLevelAPI.rst](/c:/Users/Benji/Documents/ChillProp/CoolProp/Web/coolprop/HighLevelAPI.rst:383).

`ChillProp.PropSI` currently reads fixed fluid JSON and applies no configurable reference-state shifts.

What must be added:

- A reference-state registry keyed by fluid.
- Standard presets: `IIR`, `ASHRAE`, `NBP`, `DEF`.
- Custom `(T, rhomolar, hmolar0, smolar0)` reference shifts.
- Consistent propagation into `H`, `S`, `U`, `G`, Helmholtz, and derivative outputs.

### 9. `AbstractState` parity needed to support `PropsSI`

A lot of CoolProp `PropsSI` capability depends on a more complete low-level state object. `ChillProp.AbstractState` currently supports only `PT`, `HmassP`, `SmassP`, and `QT` updates; see [chillprop/highlevel.py](/c:/Users/Benji/Documents/ChillProp/chillprop/highlevel.py:161).

What must be added underneath `PropsSI`:

- Update support for the full input-pair enum.
- Keyed output support for the full parameter enum.
- First and second partial derivative methods.
- Saturation derivative methods.
- Phase specification and phase-query methods.
- Mixture composition support.
- Reference-state handling.

## Recommended implementation order

If the goal is practical parity in stages, this is the highest-leverage order:

1. Expand input parsing and alias handling.
2. Add trivial outputs and the 2-argument overload.
3. Fill out the missing core outputs most users expect:
   - `C`, `C0`, `O`, `G`, `Z`, `Phase`, `Prandtl`
4. Implement the remaining common flash pairs:
   - `P,Q`
   - `P,D`
   - `P,U`
   - `T,U`
   - `H,S`
   - `D,H`
   - `D,S`
   - `D,U`
5. Add imposed-phase parsing.
6. Add derivative-string support.
7. Add mixture and backend string support.
8. Add reference-state compatibility.

## Bottom line

`ChillProp.PropSI` is currently a narrow CoolProp-like convenience wrapper, not a feature-complete `PropsSI` implementation.

To reach feature parity with CoolProp, `ChillProp` would need:

- the full CoolProp input-pair matrix
- a much larger output-key and alias table
- trivial property handling and the Python two-argument overload
- imposed-phase parsing
- derivative-string parsing and evaluation
- phase-query support
- backend and mixture string parsing
- reference-state support
- a substantially more complete `AbstractState` beneath `PropsSI`

The largest work items are not string parsing. They are the missing flash solvers, derivative machinery, mixture/backend support, and the low-level state model needed to make those features reliable.
