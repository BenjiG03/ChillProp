Overview
========

ChillProp provides thermodynamic and transport properties through a high-level API modeled after CoolProp for a curated set of pure and pseudo-pure fluids. The implementation uses Helmholtz-energy equations of state and JAX-native differentiation so the same public calls can support scalar evaluation, vectorized execution, and gradient-based workflows.

Current Catalog
---------------

The bundled runtime catalog is driven directly from the JSON definitions shipped in ``src/chillprop/data``. That catalog now spans permanent gases, hydrocarbons, siloxanes, solvents, legacy and modern refrigerants, cryogenic hydrogen/deuterium variants, and other HEOS pure or pseudo-pure fluids imported from CoolProp.

The authoritative supported-fluid list is maintained in ``tests/fluid_catalog.py`` by iterating over the bundled JSON payloads, and the published validation tables and plot index are regenerated from the same list in :doc:`validation`.

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
* ``docs``: Sphinx documentation sources, generated validation plots, and example assets.
* ``docs/wiki/validation_stats.json``: machine-readable parity summary used by the documentation refresh workflow.

Next Steps
----------

Proceed through :doc:`installation` and :doc:`quickstart` for the primary user path, then use :doc:`high_level_api`, :doc:`low_level_api`, :doc:`compatibility`, and :doc:`validation` to understand the current public contract and documented parity scope.
