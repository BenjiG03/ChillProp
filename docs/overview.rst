Overview
========

ChillProp provides thermodynamic and transport properties through a high-level API modeled after CoolProp for a curated set of pure and pseudo-pure fluids. The implementation uses Helmholtz-energy equations of state and JAX-native differentiation so the same public calls can support scalar evaluation, vectorized execution, and gradient-based workflows.

Project Positioning
-------------------

ChillProp is not a full drop-in replacement for CoolProp. The library currently targets a focused compatibility subset with explicit limits around backends, mixtures, derivative-string outputs, and input-pair coverage. The goal of the public documentation is to make that boundary clear while still presenting the package as a production-style technical project.

Primary Use Cases
-----------------

* Evaluate thermodynamic properties from supported state specifications.
* Run batched property calculations with ``jax.numpy`` arrays.
* Differentiate supported property calls with JAX transforms.
* Use a low-level ``AbstractState`` interface when repeated state updates are more natural than repeated high-level calls.

Repository Layout
-----------------

* ``src/chillprop``: package source.
* ``src/chillprop/data``: bundled fluid JSON files used at runtime.
* ``tests``: parity, solver, gradient, and API regression coverage.
* ``docs``: Sphinx documentation sources, plots, and example assets.

Next Steps
----------

Proceed through :doc:`installation` and :doc:`quickstart` for the primary user path, then use :doc:`high_level_api`, :doc:`low_level_api`, and :doc:`compatibility` to understand the current public contract.
