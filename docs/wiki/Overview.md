# Documentation Overview

This documentation set is organized for two use cases:

- API users who need installation guidance, public API examples, and validation scope
- Contributors who need implementation details, module boundaries, and current compatibility limits

## Current Catalog

The bundled runtime catalog currently includes `44` supported fluids spanning:

- Air and permanent gases such as `Air`, `Helium`, `Hydrogen`, `Nitrogen`, `Oxygen`, `Neon`, `Argon`, `Krypton`, and `Xenon`
- Hydrocarbons and related organics such as `Methane`, `Ethane`, `Propane`, `n-Pentane`, `n-Decane`, `n-Undecane`, `n-Dodecane`, `Cyclopentane`, `Isopentane`, `Neopentane`, `Ethanol`, and `Methanol`
- Refrigerants and specialty fluids such as `R32`, `R134a`, `R1234yf`, `R1234ze(E)`, `R404A`, `R407C`, and `R410A`

The authoritative supported-fluid list is maintained in `tests/fluid_catalog.py` and mirrored in [Validation](Validation.md).

## Reading Order

1. [Usage Guide notebook](../notebooks/Usage_Guide.ipynb)
2. [Architecture](Architecture.md)
3. [Validation](Validation.md)
4. [Implementation Gaps](Implementation_Gaps.md)

## Repository Layout

- `src/chillprop`: package source
- `src/chillprop/data`: bundled fluid JSON files used at runtime
- `tests`: automated parity, solver, gradient, and API coverage
- `docs/wiki`: prose documentation
- `docs/notebooks`: notebook-based usage material
- `docs/plots`: generated parity figures used by the documentation
- `docs/wiki/validation_stats.json`: machine-readable parity summary used by the documentation refresh workflow

## Contributor Quick Start

```bash
pip install -e .
pytest -q
```

Recommended entry points for code reading:

- [src/chillprop/highlevel.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/highlevel.py)
- [src/chillprop/parameters.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/parameters.py)
- [src/chillprop/heos.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/heos.py)
- [src/chillprop/core.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/core.py)
- [src/chillprop/solver.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/solver.py)
- [src/chillprop/transport.py](C:/Users/Benji/Documents/ChillProp/src/chillprop/transport.py)
