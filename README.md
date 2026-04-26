# ChillProp

ChillProp is a JAX-based thermodynamic property library with a CoolProp-like public API for supported pure and pseudo-pure fluids. The implementation uses Helmholtz-energy formulations, differentiable Newton solvers, and JAX transformations for automatic differentiation and batched execution.

## Status

The repository is under active development. The current implementation focuses on:

- Pure and pseudo-pure fluids bundled in `src/chillprop/data`
- `PropsSI`, `Props1SI`, `PhaseSI`, and a subset of `AbstractState`
- Differentiable state solves for `PT`, `PH`, `PS`, `TQ`, and density-temperature inputs
- Automated parity coverage against CoolProp for the supported test matrix

## Installation

```bash
pip install chillprop
```

For local development:

```bash
git clone https://github.com/BenjiG03/ChillProp.git
cd ChillProp
pip install -e .
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import chillprop.highlevel as CH

rho = CH.PropsSI("D", "T", 300.0, "P", 1e6, "Air")

T = jnp.array([280.0, 300.0, 320.0])
P = jnp.array([5e5, 1e6, 2e6])
h = CH.PropsSI("H", "T", T, "P", P, "Nitrogen")

dh_dP = jax.grad(lambda p: CH.PropsSI("H", "T", 400.0, "P", p, "Nitrogen"))(5e6)

state = CH.AbstractState("HEOS", "Nitrogen")
state.update(CH.PT_INPUTS, 1e6, 300.0)
mu = state.viscosity()
```

## Documentation

- [Documentation overview](docs/wiki/Overview.md)
- [Usage guide notebook](docs/notebooks/Usage_Guide.ipynb)
- [Architecture and implementation details](docs/wiki/Architecture.md)
- [Validation summary and parity plots](docs/wiki/Validation.md)
- [Implementation gaps relative to CoolProp](docs/wiki/Implementation_Gaps.md)

## Supported Materials

The automated high-level parity suite currently covers:

- `Air`
- `Argon`
- `CarbonDioxide`
- `Ethane`
- `Hydrogen`
- `IsoButane`
- `Methane`
- `n-Butane`
- `n-Dodecane`
- `Nitrogen`
- `Oxygen`
- `Propane`
- `Water`

Transport-property parity is exercised in the automated grid for `Argon`, `Hydrogen`, `Nitrogen`, `Oxygen`, and `Propane`.

## License

Apache 2.0
