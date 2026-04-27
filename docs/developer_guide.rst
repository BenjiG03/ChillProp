Developer Guide
===============

This page summarizes the internal structure relevant to contributors and documents how to maintain the published documentation.

Architecture Summary
--------------------

ChillProp is structured as a Helmholtz-energy property engine with a thin CoolProp-like compatibility layer. The implementation centers on typed fluid metadata, functional state evaluation, differentiable Newton solves, and a public API adapter.

Module Responsibilities
-----------------------

* ``parameters.py`` defines the typed runtime data model and parses bundled fluid JSON.
* ``heos.py`` evaluates ideal-gas and residual Helmholtz terms.
* ``core.py`` converts Helmholtz derivatives into thermodynamic properties.
* ``phases.py`` handles ancillaries, coarse phase classification, and saturation helpers.
* ``solver.py`` implements differentiable state inversion.
* ``transport.py`` evaluates viscosity and conductivity models.
* ``highlevel.py`` defines the public compatibility layer and low-level ``AbstractState`` wrapper.

Data Flow
---------

For a typical ``PropsSI(output, key1, val1, key2, val2, fluid)`` call:

#. ``highlevel.get_params()`` loads and caches the fluid definition.
#. ``highlevel._solve_state()`` selects the state inversion path for the input pair.
#. A solver returns ``(rho, T)`` in molar units.
#. ``highlevel._evaluate_output()`` dispatches to thermodynamic, phase, or transport logic.

Contributor Workflow
--------------------

When extending the public compatibility boundary:

* update the implementation in ``src/chillprop``
* add or extend regression coverage in ``tests``
* update :doc:`compatibility` when the support matrix changes
* update :doc:`validation` when the documented regression scope changes

Build Documentation
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

The Read the Docs configuration is defined in ``.readthedocs.yaml``.
