# ChillProp

ChillProp is a JAX-based thermodynamic property library with a CoolProp-like API for supported pure and pseudo-pure fluids.

## Install

```bash
pip install chillprop
```

For local development:

```bash
git clone https://github.com/BenjiG03/ChillProp.git
cd ChillProp
pip install -e .
```

## Quick Example

```python
import chillprop.highlevel as CH

rho = CH.PropsSI("D", "T", 300.0, "P", 1e6, "Air")
```

## Documentation

The full documentation site is built with Sphinx under `docs/`.

- Build locally with `sphinx-build -b html docs docs/_build/html`
- Read the landing page at `docs/_build/html/index.html`

The documentation covers installation, quickstart usage, high-level and low-level APIs, compatibility limits, validation scope, and contributor guidance.

## Status

The current implementation focuses on:

- bundled pure and pseudo-pure fluids in `src/chillprop/data`
- `PropsSI`, `Props1SI`, `PhaseSI`, and a subset of `AbstractState`
- differentiable state solves for the documented supported input pairs
- automated parity coverage against CoolProp for the documented validation matrix

## License

Apache 2.0
