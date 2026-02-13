# ChillProp ❄️🚀

**ChillProp** is a differentiable, GPU-accelerated thermodynamic property library for Python, built on [JAX](https://github.com/google/jax) and designed to be API-compatible with [CoolProp](http://www.coolprop.org/).

> [!WARNING]
> **UNDER CONSTRUCTION**: This library is currently in active development. Features and APIs may change.

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
# Will be available through PyPI in the near future
git clone https://github.com/BenjiG03/ChillProp.git
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

- **Air Components**: Nitrogen (N2), Oxygen (O2), Argon (Ar)
- **Combustion Species**: Hydrogen (H2), Carbon Dioxide (CO2), Water (H2O), Methane (CH4), Ethane (C2H6), Propane (C3H8), n-Butane, IsoButane, n-Dodecane

## Known Limitations
- **Ethane Viscosity**: The viscosity model for Ethane (Modified Batschinski-Hildebrand) is currently unstable in some regimes, yielding NaN or infinite values.
- **Complex Hydrocarbons**: Transport properties for higher hydrocarbons (n-Butane, Dodecane) may have lower accuracy at high pressures due to simplified model implementations.

## Validation & Parity

ChillProp has been rigorously validated against CoolProp for **Nitrogen**, **Air**, and **Combustion Species**:

- **Density:** Exact matches in single-phase regions.
- **VLE:** Correct phase equilibrium and saturation properties.
- **Transport:** High-accuracy viscosity and thermal conductivity models implemented for key species (N2, O2, Ar, CO2, H2, CH4).

See the [Wiki](Wiki.md) for detailed validation plots and benchmark results.

## Benchmarks

Calculations are **4.4x faster** than CoolProp when batched and JIT-compiled on CPU (AMD Ryzen 7). GPU acceleration benchmarks are pending testing.

## License

Apache 2.0
