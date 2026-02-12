# ChillProp ❄️🚀

**ChillProp** is a differentiable, GPU-accelerated thermodynamic property library for Python, built on [JAX](https://github.com/google/jax) and designed to be API-compatible with [CoolProp](http://www.coolprop.org/).

## Why ChillProp?

| Feature | CoolProp | ChillProp |
| :--- | :---: | :---: |
| **Accuracy** | ✅ Reference Standard | ✅ Matches CoolProp |
| **Differentiability** | ❌ No | ✅ **Auto-Diff (JAX)** |
| **Acceleration** | ❌ CPU Only | ✅ **GPU/TPU (JIT)** |
| **Batching** | ❌ Serial Loops | ✅ **Vectorized (vmap)** |

ChillProp allows you to incorporate accurate thermodynamic properties directly into your machine learning models, optimization loops, and differentiable physics simulations.

## Installation

```bash
git clone https://github.com/google-deepmind/ChillProp.git
cd ChillProp
pip install -e .
```

## Quick Start

ChillProp mimics the CoolProp High-Level API (`PropsSI`, `AbstractState`).

### using `PropsSI`

```python
import chillprop.highlevel as CH

# Density of Nitrogen at 300 K, 10 bar
rho = CH.PropsSI("D", "T", 300, "P", 1e6, "Nitrogen")
print(f"Density: {rho} kg/m3")
```

### using `AbstractState`

```python
import chillprop.highlevel as CH
import jax.numpy as jnp

# Create state
AS = CH.AbstractState("HEOS", "Nitrogen")

# Update state (PT Input)
AS.update(CH.PT_INPUTS, 1e6, 300)

print(f"Viscosity: {AS.viscosity()} Pa-s")
print(f"Conductivity: {AS.conductivity()} W/m/K")
```

## Validation & Parity

ChillProp has been rigorously validated against CoolProp for **Nitrogen**:

- **Density:** Exact matches in single-phase regions.
- **VLE:** Correct phase equilibrium and saturation properties.
- **Transport:** High-accuracy viscosity and thermal conductivity models implemented.

See the [Wiki](Wiki.md) for detailed validation plots and benchmark results.

## Benchmarks

Calculations are **4.4x faster** than CoolProp when batched and JIT-compiled on CPU (Intel i9). GPU acceleration offers further speedups for massive batches.

## License

Apache 2.0
