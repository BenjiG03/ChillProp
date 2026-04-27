ChillProp
=========

ChillProp is a JAX-based thermodynamic property library with a CoolProp-like API for supported pure and pseudo-pure fluids. The implementation is built around Helmholtz-energy formulations, differentiable state solves, and JAX transformations for batched and gradient-based workflows.

The documentation is organized for two audiences:

* Users who need installation guidance, examples, API behavior, compatibility limits, and validation scope.
* Contributors who need architecture notes, implementation boundaries, and local documentation build instructions.

Getting Started
---------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   overview
   installation
   quickstart
   examples
   high_level_api
   low_level_api
   compatibility
   validation

Project Internals
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide
   api_reference

Highlights
----------

* CoolProp-like entry points for ``PropsSI``, ``Props1SI``, ``PhaseSI``, and a subset of ``AbstractState``.
* JAX-compatible scalar, batched, and autodiff-oriented property evaluation for supported input pairs.
* Bundled fluid metadata with no external CoolProp checkout required at runtime.
* Regression coverage and parity plots that document the currently supported subset explicitly.

Documentation Conventions
-------------------------

* Public examples use the current supported API subset only.
* Compatibility notes describe the implemented boundary relative to CoolProp.
* Validation pages distinguish between behavior that is implemented and behavior that is covered by automated regression tests.
