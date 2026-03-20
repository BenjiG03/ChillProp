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
| **Batching** | ❌ Serial Loops | ✅ **Vectorized (Native Arrays)** |

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
import jax.numpy as jnp

# Scalar evaluation (Density of Air at 300 K, 10 bar)
rho = CH.PropsSI("D", "T", 300.0, "P", 1e6, "Air")
print(f"Density: {rho} kg/m3")

# Vectorized evaluation (Zero-overhead batching)
T_vec = jnp.array([300.0, 310.0, 320.0])
P_vec = jnp.array([1e6, 1e6, 1.5e6])
rho_vec = CH.PropsSI("D", "T", T_vec, "P", P_vec, "Air")
print(f"Vector Density: {rho_vec}")
```

### Automatic Differentiation (`jax.grad`)

ChillProp is natively differentiable! You can extract arbitrary thermodynamic partial derivatives seamlessly using JAX, resolving mathematical derivatives rather than slow or unstable finite differences:

```python
import jax
import chillprop.highlevel as CH

# Target function: Calculate Enthalpy given T and P
def calc_enthalpy(P):
    # Returns scalar enthalpy for Nitrogen at 400 K
    return CH.PropsSI("H", "T", 400.0, "P", P, "Nitrogen")

# Create a compiled derivative function: dh/dP
dh_dP_func = jax.grad(calc_enthalpy)

# Evaluate the exact derivative at P = 5 MPa
exact_derivative = dh_dP_func(5e6)
print(f"dh/dP at 5 MPa: {exact_derivative} J/kg/Pa")
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
