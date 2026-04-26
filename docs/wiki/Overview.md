# Documentation Overview

This documentation set is organized for two use cases:

- API users who need installation guidance, public API examples, and validation scope
- Contributors who need implementation details, module boundaries, and current compatibility limits

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
