High-Level API
==============

The primary entry point is ``chillprop.highlevel``. The public interface mirrors a subset of CoolProp's string-based API while preserving JAX-friendly execution.

Core Functions
--------------

``PropsSI``
^^^^^^^^^^^

``PropsSI`` supports:

* Trivial constant queries with ``PropsSI(output, fluid)``.
* State-dependent property queries with ``PropsSI(output, key1, value1, key2, value2, fluid)``.
* Scalar and array-like inputs for supported state specifications.

Supported Input Pairs
---------------------

The high-level API currently supports these state specifications:

* ``P`` and ``T``
* ``P`` and ``H``
* ``P`` and ``S``
* ``T`` and ``Q``
* ``T`` and ``D``
* ``T`` and ``Dmolar``

Unsupported input pairs raise ``NotImplementedError``.

Supported Output Categories
---------------------------

* Trivial fluid constants such as critical, triple-point, and molar-mass properties.
* Core thermodynamic properties including density, enthalpy, entropy, internal energy, heat capacities, pressure, and compressibility factor.
* Phase helpers including ``Phase`` and ``PhaseSI``.
* Transport outputs required by the documented validation scope, including viscosity, conductivity, and Prandtl number where supported by the fluid model.

Units and Conventions
---------------------

* Internal density is molar density in ``mol/m^3``.
* Internal energy-like properties are primarily computed on a molar basis.
* Mass-basis aliases are handled at the public API boundary.
* Reduced variables follow the usual Helmholtz convention ``tau = Tr / T`` and ``delta = rho / rhor``.

Public Limitations
------------------

* Mixture strings are not implemented.
* Derivative-string outputs such as ``d(Hmass)/d(P)|T`` are not implemented.
* Phase-imposed input keys such as ``T|liquid`` are not implemented.
* Mutable reference-state APIs are not implemented.

API Reference
-------------

See :doc:`api_reference` for generated signatures and member documentation for the public functions and constants.
