# Validation

This page summarizes the current automated parity coverage exercised against CoolProp for the documented ChillProp fluid catalog. The statistics and plot gallery on this page are derived from the same grids used by `tests/test_highlevel_pure_jax.py`.

## Catalog Snapshot

- Supported runtime fluids: `124`
- Automated transport subset: `6` fluids
- Automated two-phase subset: `80` fluids
- Supported fluids: `1-Butene`, `Acetone`, `Air`, `Ammonia`, `Argon`, `Benzene`, `CarbonDioxide`, `CarbonMonoxide`, `CarbonylSulfide`, `cis-2-Butene`, `CycloHexane`, `Cyclopentane`, `CycloPropane`, `D4`, `D5`, `D6`, `Deuterium`, `Dichloroethane`, `DiethylEther`, `DimethylCarbonate`, `DimethylEther`, `Ethane`, `Ethanol`, `EthylBenzene`, `Ethylene`, `EthyleneOxide`, `Fluorine`, `HeavyWater`, `Helium`, `HFE143m`, `Hydrogen`, `HydrogenChloride`, `HydrogenSulfide`, `IsoButane`, `IsoButene`, `Isohexane`, `Isopentane`, `Krypton`, `m-Xylene`, `MD2M`, `MD3M`, `MD4M`, `MDM`, `Methane`, `Methanol`, `MethylLinoleate`, `MethylLinolenate`, `MethylOleate`, `MethylPalmitate`, `MethylStearate`, `MM`, `n-Butane`, `n-Decane`, `n-Dodecane`, `n-Heptane`, `n-Hexane`, `n-Nonane`, `n-Octane`, `n-Pentane`, `n-Propane`, `n-Undecane`, `Neon`, `Neopentane`, `Nitrogen`, `NitrousOxide`, `Novec649`, `o-Xylene`, `OrthoDeuterium`, `OrthoHydrogen`, `Oxygen`, `p-Xylene`, `ParaDeuterium`, `ParaHydrogen`, `Propylene`, `Propyne`, `R11`, `R113`, `R114`, `R115`, `R116`, `R12`, `R123`, `R1233zd(E)`, `R1234yf`, `R1234ze(E)`, `R1234ze(Z)`, `R124`, `R1243zf`, `R125`, `R13`, `R1336mzz(E)`, `R134a`, `R13I1`, `R14`, `R141b`, `R142b`, `R143a`, `R152A`, `R161`, `R21`, `R218`, `R22`, `R227EA`, `R23`, `R236EA`, `R236FA`, `R245ca`, `R245fa`, `R32`, `R365MFC`, `R40`, `R404A`, `R407C`, `R41`, `R410A`, `R507A`, `RC318`, `SES36`, `SulfurDioxide`, `SulfurHexafluoride`, `Toluene`, `trans-2-Butene`, `Water`, `Xenon`
- Transport subset: `Argon`, `Hydrogen`, `n-Decane`, `Nitrogen`, `Oxygen`, `Propane`
- Two-phase subset: `1-Butene`, `Acetone`, `Ammonia`, `Argon`, `Benzene`, `CarbonDioxide`, `CarbonMonoxide`, `CarbonylSulfide`, `cis-2-Butene`, `CycloHexane`, `Cyclopentane`, `Deuterium`, `DimethylCarbonate`, `DimethylEther`, `Ethane`, `EthylBenzene`, `Ethylene`, `Fluorine`, `HeavyWater`, `Helium`, `HydrogenChloride`, `IsoButane`, `IsoButene`, `Isopentane`, `m-Xylene`, `MD2M`, `MD4M`, `MDM`, `Methane`, `MethylLinoleate`, `MethylLinolenate`, `MethylOleate`, `MethylPalmitate`, `MethylStearate`, `MM`, `n-Butane`, `n-Decane`, `n-Octane`, `n-Pentane`, `n-Propane`, `n-Undecane`, `Neon`, `Neopentane`, `Nitrogen`, `Novec649`, `o-Xylene`, `OrthoDeuterium`, `OrthoHydrogen`, `p-Xylene`, `ParaDeuterium`, `Propylene`, `R11`, `R113`, `R115`, `R116`, `R12`, `R123`, `R1233zd(E)`, `R1234yf`, `R1234ze(Z)`, `R1336mzz(E)`, `R13I1`, `R141b`, `R142b`, `R143a`, `R152A`, `R218`, `R22`, `R227EA`, `R23`, `R236FA`, `R245ca`, `R32`, `RC318`, `SulfurDioxide`, `SulfurHexafluoride`, `Toluene`, `trans-2-Butene`, `Water`, `Xenon`

## Validation Scope

- Trivial fluid constants: 11 checks per fluid
- Single-phase core parity grid: 4 state points x 20 outputs per supported fluid
- Transport parity grid: 4 state points x 3 outputs for the dedicated transport subset
- Two-phase parity grid: 6 state points x 10 outputs for the documented two-phase subset

## Tolerances

| Category | Default tolerance | Notes |
| :--- | ---: | :--- |
| Trivial outputs | `3e-2` | Scalar constants such as critical properties and molar mass |
| Single-phase core outputs | `5e-8` | Property-specific relaxations are applied in the test suite |
| Two-phase outputs | `1e-3` | `Q` uses `1e-4`; only the documented subset is enforced |
| Transport outputs | `5e-3` | Viscosity, conductivity, and Prandtl number for the transport subset |

## Supported Materials Summary

| Fluid | Trivial max rel err | Single-phase max rel err | Transport max rel err | Two-phase max rel err | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| ``1-Butene`` | ``2.441e-15`` | ``3.082e-11`` | ``n/a`` | ``9.928e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Acetone`` | ``9.924e-16`` | ``3.561e-12`` | ``n/a`` | ``8.194e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Air`` | ``1.820e-04`` | ``2.357e-08`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``Ammonia`` | ``4.917e-16`` | ``1.259e-10`` | ``n/a`` | ``3.891e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Argon`` | ``7.660e-16`` | ``7.508e-09`` | ``4.033e-08`` | ``7.818e-12`` | Full documented suite for current subsets |
| ``Benzene`` | ``3.417e-15`` | ``3.040e-10`` | ``n/a`` | ``6.056e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``CarbonDioxide`` | ``4.166e-15`` | ``2.871e-11`` | ``n/a`` | ``5.989e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``CarbonMonoxide`` | ``2.662e-15`` | ``1.486e-09`` | ``n/a`` | ``2.206e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``CarbonylSulfide`` | ``2.632e-15`` | ``2.209e-11`` | ``n/a`` | ``6.090e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``cis-2-Butene`` | ``1.099e-15`` | ``1.645e-12`` | ``n/a`` | ``1.178e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``CycloHexane`` | ``9.129e-16`` | ``4.596e-07`` | ``n/a`` | ``1.412e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Cyclopentane`` | ``2.439e-15`` | ``6.513e-11`` | ``n/a`` | ``2.128e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``CycloPropane`` | ``4.154e-15`` | ``2.179e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``D4`` | ``3.456e-16`` | ``3.608e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``D5`` | ``4.753e-15`` | ``6.720e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``D6`` | ``3.390e-15`` | ``8.573e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Deuterium`` | ``3.466e-15`` | ``9.360e-09`` | ``n/a`` | ``2.361e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Dichloroethane`` | ``1.185e-16`` | ``5.700e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``DiethylEther`` | ``1.628e-15`` | ``2.982e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``DimethylCarbonate`` | ``2.846e-15`` | ``3.942e-11`` | ``n/a`` | ``2.547e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``DimethylEther`` | ``3.665e-15`` | ``1.751e-10`` | ``n/a`` | ``1.132e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Ethane`` | ``1.720e-15`` | ``7.822e-09`` | ``n/a`` | ``2.521e-11`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Ethanol`` | ``7.429e-16`` | ``7.158e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``EthylBenzene`` | ``1.543e-15`` | ``1.248e-10`` | ``n/a`` | ``1.242e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Ethylene`` | ``3.694e-16`` | ``4.154e-12`` | ``n/a`` | ``1.424e-10`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``EthyleneOxide`` | ``3.825e-16`` | ``1.326e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Fluorine`` | ``7.110e-15`` | ``5.355e-09`` | ``n/a`` | ``4.294e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``HeavyWater`` | ``1.049e-14`` | ``3.679e-12`` | ``n/a`` | ``1.092e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Helium`` | ``2.039e-15`` | ``5.620e-10`` | ``n/a`` | ``5.561e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``HFE143m`` | ``3.960e-15`` | ``9.002e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Hydrogen`` | ``3.592e-16`` | ``1.276e-10`` | ``2.959e-05`` | ``n/a`` | This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``HydrogenChloride`` | ``3.361e-15`` | ``2.487e-09`` | ``n/a`` | ``1.147e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``HydrogenSulfide`` | ``2.070e-15`` | ``1.886e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``IsoButane`` | ``0.000e+00`` | ``7.918e-11`` | ``n/a`` | ``5.327e-11`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``IsoButene`` | ``4.059e-15`` | ``9.664e-11`` | ``n/a`` | ``5.097e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Isohexane`` | ``1.377e-15`` | ``1.744e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Isopentane`` | ``0.000e+00`` | ``2.594e-11`` | ``n/a`` | ``3.510e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Krypton`` | ``1.348e-15`` | ``7.662e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``m-Xylene`` | ``1.186e-15`` | ``1.360e-10`` | ``n/a`` | ``6.592e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MD2M`` | ``6.106e-16`` | ``7.122e-10`` | ``n/a`` | ``1.610e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MD3M`` | ``2.563e-15`` | ``4.881e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``MD4M`` | ``2.810e-15`` | ``1.524e-09`` | ``n/a`` | ``4.338e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MDM`` | ``1.296e-15`` | ``1.021e-09`` | ``n/a`` | ``8.588e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Methane`` | ``1.620e-15`` | ``1.720e-09`` | ``n/a`` | ``1.644e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Methanol`` | ``2.607e-15`` | ``2.341e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``MethylLinoleate`` | ``4.688e-15`` | ``6.071e-12`` | ``n/a`` | ``1.474e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MethylLinolenate`` | ``1.701e-16`` | ``2.591e-10`` | ``n/a`` | ``2.949e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MethylOleate`` | ``1.682e-15`` | ``8.063e-10`` | ``n/a`` | ``1.235e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MethylPalmitate`` | ``6.899e-16`` | ``2.593e-10`` | ``n/a`` | ``6.924e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MethylStearate`` | ``2.067e-15`` | ``2.228e-10`` | ``n/a`` | ``6.870e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``MM`` | ``5.426e-15`` | ``1.116e-10`` | ``n/a`` | ``5.002e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``n-Butane`` | ``7.360e-16`` | ``2.890e-09`` | ``n/a`` | ``8.235e-12`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``n-Decane`` | ``2.216e-15`` | ``1.172e-12`` | ``1.839e-05`` | ``8.743e-05`` | Full documented suite for current subsets |
| ``n-Dodecane`` | ``1.153e-15`` | ``1.604e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``n-Heptane`` | ``2.182e-15`` | ``1.880e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``n-Hexane`` | ``5.354e-15`` | ``6.989e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``n-Nonane`` | ``1.428e-15`` | ``2.974e-08`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``n-Octane`` | ``1.500e-15`` | ``1.062e-10`` | ``n/a`` | ``4.486e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``n-Pentane`` | ``6.914e-15`` | ``3.354e-10`` | ``n/a`` | ``1.320e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``n-Propane`` | ``1.972e-15`` | ``4.451e-12`` | ``n/a`` | ``8.312e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``n-Undecane`` | ``2.924e-15`` | ``2.182e-11`` | ``n/a`` | ``1.728e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Neon`` | ``9.972e-15`` | ``1.355e-11`` | ``n/a`` | ``5.243e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Neopentane`` | ``1.748e-15`` | ``2.436e-11`` | ``n/a`` | ``2.213e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Nitrogen`` | ``1.097e-15`` | ``3.271e-09`` | ``3.339e-08`` | ``5.616e-09`` | Full documented suite for current subsets |
| ``NitrousOxide`` | ``2.571e-15`` | ``2.225e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Novec649`` | ``2.616e-15`` | ``1.429e-10`` | ``n/a`` | ``2.952e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``o-Xylene`` | ``3.987e-15`` | ``1.145e-10`` | ``n/a`` | ``1.203e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``OrthoDeuterium`` | ``3.466e-15`` | ``9.281e-09`` | ``n/a`` | ``2.361e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``OrthoHydrogen`` | ``5.333e-16`` | ``6.087e-11`` | ``n/a`` | ``9.188e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Oxygen`` | ``2.030e-15`` | ``4.389e-09`` | ``1.300e-03`` | ``n/a`` | This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``p-Xylene`` | ``1.450e-15`` | ``1.132e-10`` | ``n/a`` | ``3.878e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``ParaDeuterium`` | ``3.466e-15`` | ``9.525e-09`` | ``n/a`` | ``2.361e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``ParaHydrogen`` | ``2.173e-15`` | ``6.891e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``Propylene`` | ``1.636e-15`` | ``1.392e-11`` | ``n/a`` | ``1.763e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Propyne`` | ``4.445e-15`` | ``8.596e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R11`` | ``6.339e-16`` | ``3.403e-11`` | ``n/a`` | ``6.878e-11`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R113`` | ``2.334e-15`` | ``2.873e-11`` | ``n/a`` | ``4.233e-08`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R114`` | ``5.556e-16`` | ``7.281e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R115`` | ``1.488e-15`` | ``2.601e-11`` | ``n/a`` | ``8.086e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R116`` | ``6.112e-16`` | ``1.225e-11`` | ``n/a`` | ``1.076e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R12`` | ``2.477e-15`` | ``9.573e-10`` | ``n/a`` | ``3.006e-08`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R123`` | ``8.139e-15`` | ``3.336e-09`` | ``n/a`` | ``7.164e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R1233zd(E)`` | ``3.341e-15`` | ``5.518e-11`` | ``n/a`` | ``1.040e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R1234yf`` | ``3.580e-15`` | ``4.442e-11`` | ``n/a`` | ``4.778e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R1234ze(E)`` | ``3.843e-16`` | ``1.765e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R1234ze(Z)`` | ``4.221e-15`` | ``3.891e-11`` | ``n/a`` | ``3.512e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R124`` | ``3.469e-15`` | ``6.448e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R1243zf`` | ``3.177e-15`` | ``7.822e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R125`` | ``4.504e-15`` | ``7.792e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R13`` | ``1.172e-16`` | ``1.324e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R1336mzz(E)`` | ``3.854e-15`` | ``7.947e-11`` | ``n/a`` | ``5.067e-11`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R134a`` | ``6.883e-16`` | ``9.496e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R13I1`` | ``1.178e-15`` | ``4.016e-12`` | ``n/a`` | ``6.840e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R14`` | ``1.361e-15`` | ``1.486e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R141b`` | ``8.845e-16`` | ``3.764e-11`` | ``n/a`` | ``7.962e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R142b`` | ``2.641e-15`` | ``3.337e-13`` | ``n/a`` | ``6.507e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R143a`` | ``1.485e-15`` | ``6.152e-11`` | ``n/a`` | ``1.906e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R152A`` | ``1.588e-14`` | ``1.530e-13`` | ``n/a`` | ``1.639e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R161`` | ``4.276e-15`` | ``4.818e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R21`` | ``4.755e-15`` | ``1.155e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R218`` | ``1.940e-15`` | ``2.446e-11`` | ``n/a`` | ``6.818e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R22`` | ``4.797e-14`` | ``8.878e-11`` | ``n/a`` | ``5.891e-09`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R227EA`` | ``3.184e-16`` | ``1.362e-10`` | ``n/a`` | ``3.007e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R23`` | ``4.241e-15`` | ``1.640e-10`` | ``n/a`` | ``9.805e-05`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R236EA`` | ``2.319e-15`` | ``1.505e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R236FA`` | ``1.605e-15`` | ``1.426e-12`` | ``n/a`` | ``1.441e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R245ca`` | ``3.545e-16`` | ``1.711e-12`` | ``n/a`` | ``1.382e-07`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R245fa`` | ``1.275e-16`` | ``5.637e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R32`` | ``1.611e-15`` | ``1.481e-12`` | ``n/a`` | ``1.880e-06`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``R365MFC`` | ``8.554e-16`` | ``3.073e-11`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R40`` | ``1.210e-15`` | ``1.795e-09`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R404A`` | ``1.299e-05`` | ``1.143e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``R407C`` | ``4.944e-06`` | ``2.893e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``R41`` | ``3.154e-16`` | ``2.264e-15`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance. |
| ``R410A`` | ``2.815e-06`` | ``1.780e-13`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``R507A`` | ``1.945e-05`` | ``1.616e-12`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``RC318`` | ``3.185e-15`` | ``4.081e-11`` | ``n/a`` | ``3.853e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``SES36`` | ``2.854e-03`` | ``1.192e-10`` | ``n/a`` | ``n/a`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset.; CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied. |
| ``SulfurDioxide`` | ``3.543e-16`` | ``5.541e-12`` | ``n/a`` | ``7.017e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``SulfurHexafluoride`` | ``1.364e-15`` | ``1.843e-09`` | ``n/a`` | ``1.541e-10`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Toluene`` | ``1.129e-16`` | ``8.404e-13`` | ``n/a`` | ``1.217e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``trans-2-Butene`` | ``4.634e-16`` | ``4.811e-11`` | ``n/a`` | ``6.031e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Water`` | ``3.208e-15`` | ``5.327e-11`` | ``n/a`` | ``3.483e-08`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |
| ``Xenon`` | ``3.188e-16`` | ``7.911e-09`` | ``n/a`` | ``2.534e-04`` | Transport properties may be implemented, but this fluid is outside the automated transport parity subset. |

## Plot Gallery

The complete per-fluid single-phase parity plot set is stored in `docs/plots/validated` and indexed below.

### `1-Butene`

![1-Butene parity plot](../plots/validated/1-Butene_parity.png)

### `Acetone`

![Acetone parity plot](../plots/validated/Acetone_parity.png)

### `Air`

![Air parity plot](../plots/validated/Air_parity.png)

### `Ammonia`

![Ammonia parity plot](../plots/validated/Ammonia_parity.png)

### `Argon`

![Argon parity plot](../plots/validated/Argon_parity.png)

### `Benzene`

![Benzene parity plot](../plots/validated/Benzene_parity.png)

### `CarbonDioxide`

![CarbonDioxide parity plot](../plots/validated/CarbonDioxide_parity.png)

### `CarbonMonoxide`

![CarbonMonoxide parity plot](../plots/validated/CarbonMonoxide_parity.png)

### `CarbonylSulfide`

![CarbonylSulfide parity plot](../plots/validated/CarbonylSulfide_parity.png)

### `cis-2-Butene`

![cis-2-Butene parity plot](../plots/validated/cis-2-Butene_parity.png)

### `CycloHexane`

![CycloHexane parity plot](../plots/validated/CycloHexane_parity.png)

### `Cyclopentane`

![Cyclopentane parity plot](../plots/validated/Cyclopentane_parity.png)

### `CycloPropane`

![CycloPropane parity plot](../plots/validated/CycloPropane_parity.png)

### `D4`

![D4 parity plot](../plots/validated/D4_parity.png)

### `D5`

![D5 parity plot](../plots/validated/D5_parity.png)

### `D6`

![D6 parity plot](../plots/validated/D6_parity.png)

### `Deuterium`

![Deuterium parity plot](../plots/validated/Deuterium_parity.png)

### `Dichloroethane`

![Dichloroethane parity plot](../plots/validated/Dichloroethane_parity.png)

### `DiethylEther`

![DiethylEther parity plot](../plots/validated/DiethylEther_parity.png)

### `DimethylCarbonate`

![DimethylCarbonate parity plot](../plots/validated/DimethylCarbonate_parity.png)

### `DimethylEther`

![DimethylEther parity plot](../plots/validated/DimethylEther_parity.png)

### `Ethane`

![Ethane parity plot](../plots/validated/Ethane_parity.png)

### `Ethanol`

![Ethanol parity plot](../plots/validated/Ethanol_parity.png)

### `EthylBenzene`

![EthylBenzene parity plot](../plots/validated/EthylBenzene_parity.png)

### `Ethylene`

![Ethylene parity plot](../plots/validated/Ethylene_parity.png)

### `EthyleneOxide`

![EthyleneOxide parity plot](../plots/validated/EthyleneOxide_parity.png)

### `Fluorine`

![Fluorine parity plot](../plots/validated/Fluorine_parity.png)

### `HeavyWater`

![HeavyWater parity plot](../plots/validated/HeavyWater_parity.png)

### `Helium`

![Helium parity plot](../plots/validated/Helium_parity.png)

### `HFE143m`

![HFE143m parity plot](../plots/validated/HFE143m_parity.png)

### `Hydrogen`

![Hydrogen parity plot](../plots/validated/Hydrogen_parity.png)

### `HydrogenChloride`

![HydrogenChloride parity plot](../plots/validated/HydrogenChloride_parity.png)

### `HydrogenSulfide`

![HydrogenSulfide parity plot](../plots/validated/HydrogenSulfide_parity.png)

### `IsoButane`

![IsoButane parity plot](../plots/validated/IsoButane_parity.png)

### `IsoButene`

![IsoButene parity plot](../plots/validated/IsoButene_parity.png)

### `Isohexane`

![Isohexane parity plot](../plots/validated/Isohexane_parity.png)

### `Isopentane`

![Isopentane parity plot](../plots/validated/Isopentane_parity.png)

### `Krypton`

![Krypton parity plot](../plots/validated/Krypton_parity.png)

### `m-Xylene`

![m-Xylene parity plot](../plots/validated/m-Xylene_parity.png)

### `MD2M`

![MD2M parity plot](../plots/validated/MD2M_parity.png)

### `MD3M`

![MD3M parity plot](../plots/validated/MD3M_parity.png)

### `MD4M`

![MD4M parity plot](../plots/validated/MD4M_parity.png)

### `MDM`

![MDM parity plot](../plots/validated/MDM_parity.png)

### `Methane`

![Methane parity plot](../plots/validated/Methane_parity.png)

### `Methanol`

![Methanol parity plot](../plots/validated/Methanol_parity.png)

### `MethylLinoleate`

![MethylLinoleate parity plot](../plots/validated/MethylLinoleate_parity.png)

### `MethylLinolenate`

![MethylLinolenate parity plot](../plots/validated/MethylLinolenate_parity.png)

### `MethylOleate`

![MethylOleate parity plot](../plots/validated/MethylOleate_parity.png)

### `MethylPalmitate`

![MethylPalmitate parity plot](../plots/validated/MethylPalmitate_parity.png)

### `MethylStearate`

![MethylStearate parity plot](../plots/validated/MethylStearate_parity.png)

### `MM`

![MM parity plot](../plots/validated/MM_parity.png)

### `n-Butane`

![n-Butane parity plot](../plots/validated/n-Butane_parity.png)

### `n-Decane`

![n-Decane parity plot](../plots/validated/n-Decane_parity.png)

### `n-Dodecane`

![n-Dodecane parity plot](../plots/validated/n-Dodecane_parity.png)

### `n-Heptane`

![n-Heptane parity plot](../plots/validated/n-Heptane_parity.png)

### `n-Hexane`

![n-Hexane parity plot](../plots/validated/n-Hexane_parity.png)

### `n-Nonane`

![n-Nonane parity plot](../plots/validated/n-Nonane_parity.png)

### `n-Octane`

![n-Octane parity plot](../plots/validated/n-Octane_parity.png)

### `n-Pentane`

![n-Pentane parity plot](../plots/validated/n-Pentane_parity.png)

### `n-Propane`

![n-Propane parity plot](../plots/validated/n-Propane_parity.png)

### `n-Undecane`

![n-Undecane parity plot](../plots/validated/n-Undecane_parity.png)

### `Neon`

![Neon parity plot](../plots/validated/Neon_parity.png)

### `Neopentane`

![Neopentane parity plot](../plots/validated/Neopentane_parity.png)

### `Nitrogen`

![Nitrogen parity plot](../plots/validated/Nitrogen_parity.png)

### `NitrousOxide`

![NitrousOxide parity plot](../plots/validated/NitrousOxide_parity.png)

### `Novec649`

![Novec649 parity plot](../plots/validated/Novec649_parity.png)

### `o-Xylene`

![o-Xylene parity plot](../plots/validated/o-Xylene_parity.png)

### `OrthoDeuterium`

![OrthoDeuterium parity plot](../plots/validated/OrthoDeuterium_parity.png)

### `OrthoHydrogen`

![OrthoHydrogen parity plot](../plots/validated/OrthoHydrogen_parity.png)

### `Oxygen`

![Oxygen parity plot](../plots/validated/Oxygen_parity.png)

### `p-Xylene`

![p-Xylene parity plot](../plots/validated/p-Xylene_parity.png)

### `ParaDeuterium`

![ParaDeuterium parity plot](../plots/validated/ParaDeuterium_parity.png)

### `ParaHydrogen`

![ParaHydrogen parity plot](../plots/validated/ParaHydrogen_parity.png)

### `Propylene`

![Propylene parity plot](../plots/validated/Propylene_parity.png)

### `Propyne`

![Propyne parity plot](../plots/validated/Propyne_parity.png)

### `R11`

![R11 parity plot](../plots/validated/R11_parity.png)

### `R113`

![R113 parity plot](../plots/validated/R113_parity.png)

### `R114`

![R114 parity plot](../plots/validated/R114_parity.png)

### `R115`

![R115 parity plot](../plots/validated/R115_parity.png)

### `R116`

![R116 parity plot](../plots/validated/R116_parity.png)

### `R12`

![R12 parity plot](../plots/validated/R12_parity.png)

### `R123`

![R123 parity plot](../plots/validated/R123_parity.png)

### `R1233zd(E)`

![R1233zd(E) parity plot](../plots/validated/R1233zd(E)_parity.png)

### `R1234yf`

![R1234yf parity plot](../plots/validated/R1234yf_parity.png)

### `R1234ze(E)`

![R1234ze(E) parity plot](../plots/validated/R1234ze(E)_parity.png)

### `R1234ze(Z)`

![R1234ze(Z) parity plot](../plots/validated/R1234ze(Z)_parity.png)

### `R124`

![R124 parity plot](../plots/validated/R124_parity.png)

### `R1243zf`

![R1243zf parity plot](../plots/validated/R1243zf_parity.png)

### `R125`

![R125 parity plot](../plots/validated/R125_parity.png)

### `R13`

![R13 parity plot](../plots/validated/R13_parity.png)

### `R1336mzz(E)`

![R1336mzz(E) parity plot](../plots/validated/R1336mzz(E)_parity.png)

### `R134a`

![R134a parity plot](../plots/validated/R134a_parity.png)

### `R13I1`

![R13I1 parity plot](../plots/validated/R13I1_parity.png)

### `R14`

![R14 parity plot](../plots/validated/R14_parity.png)

### `R141b`

![R141b parity plot](../plots/validated/R141b_parity.png)

### `R142b`

![R142b parity plot](../plots/validated/R142b_parity.png)

### `R143a`

![R143a parity plot](../plots/validated/R143a_parity.png)

### `R152A`

![R152A parity plot](../plots/validated/R152A_parity.png)

### `R161`

![R161 parity plot](../plots/validated/R161_parity.png)

### `R21`

![R21 parity plot](../plots/validated/R21_parity.png)

### `R218`

![R218 parity plot](../plots/validated/R218_parity.png)

### `R22`

![R22 parity plot](../plots/validated/R22_parity.png)

### `R227EA`

![R227EA parity plot](../plots/validated/R227EA_parity.png)

### `R23`

![R23 parity plot](../plots/validated/R23_parity.png)

### `R236EA`

![R236EA parity plot](../plots/validated/R236EA_parity.png)

### `R236FA`

![R236FA parity plot](../plots/validated/R236FA_parity.png)

### `R245ca`

![R245ca parity plot](../plots/validated/R245ca_parity.png)

### `R245fa`

![R245fa parity plot](../plots/validated/R245fa_parity.png)

### `R32`

![R32 parity plot](../plots/validated/R32_parity.png)

### `R365MFC`

![R365MFC parity plot](../plots/validated/R365MFC_parity.png)

### `R40`

![R40 parity plot](../plots/validated/R40_parity.png)

### `R404A`

![R404A parity plot](../plots/validated/R404A_parity.png)

### `R407C`

![R407C parity plot](../plots/validated/R407C_parity.png)

### `R41`

![R41 parity plot](../plots/validated/R41_parity.png)

### `R410A`

![R410A parity plot](../plots/validated/R410A_parity.png)

### `R507A`

![R507A parity plot](../plots/validated/R507A_parity.png)

### `RC318`

![RC318 parity plot](../plots/validated/RC318_parity.png)

### `SES36`

![SES36 parity plot](../plots/validated/SES36_parity.png)

### `SulfurDioxide`

![SulfurDioxide parity plot](../plots/validated/SulfurDioxide_parity.png)

### `SulfurHexafluoride`

![SulfurHexafluoride parity plot](../plots/validated/SulfurHexafluoride_parity.png)

### `Toluene`

![Toluene parity plot](../plots/validated/Toluene_parity.png)

### `trans-2-Butene`

![trans-2-Butene parity plot](../plots/validated/trans-2-Butene_parity.png)

### `Water`

![Water parity plot](../plots/validated/Water_parity.png)

### `Xenon`

![Xenon parity plot](../plots/validated/Xenon_parity.png)
