Supported Fluids and Current Limits
===================================

ChillProp documents a curated compatibility subset rather than full CoolProp feature parity.

Backend Coverage
----------------

Supported:

* default backend
* ``HEOS``

Not implemented:

* ``REFPROP``
* incompressible backends
* tabular backends
* cubic EOS backends
* other backend-specific extensions exposed by CoolProp

Fluid Coverage
--------------

Supported:

* bundled pure and pseudo-pure fluids shipped in ``src/chillprop/data``

Not implemented:

* mixture strings such as ``HEOS::Propane[0.5]&Ethane[0.5]``
* runtime discovery of the full CoolProp fluid catalog

Input-Pair Coverage
-------------------

Implemented high-level state specifications:

* ``P``, ``T``
* ``P``, ``H``
* ``P``, ``S``
* ``T``, ``Q``
* ``T``, ``D``
* ``T``, ``Dmolar``

Not implemented:

* the broader CoolProp matrix such as ``PU``, ``HS``, ``DQ``, and related pairs
* phase-imposed input keys such as ``T|liquid``

Output Coverage
---------------

Implemented output groups:

* trivial fluid constants
* core thermodynamic properties
* phase index and phase string helpers
* transport outputs required by the current public validation scope

Not implemented:

* derivative-string outputs such as ``d(Hmass)/d(P)|T``
* the full keyed-output catalog exposed by CoolProp
* every convenience alias accepted by CoolProp

``AbstractState`` Coverage
--------------------------

Implemented:

* construction with supported backends and fluids
* ``update()`` for the supported input-pair subset
* common scalar accessors
* a subset of keyed outputs

Not implemented:

* the full mutable ``AbstractState`` surface
* general phase-envelope tooling
* comprehensive mixture state operations

Reference-State Mutation
------------------------

``set_reference_state(...)`` raises ``NotImplementedError``. Reference-state behavior is fixed by the bundled fluid definitions.

Transport Validation Boundary
-----------------------------

Transport-model implementations exist for more fluids than the current automated transport regression grid. The documented transport parity suite currently covers:

* ``Argon``
* ``Hydrogen``
* ``Nitrogen``
* ``Oxygen``
* ``Propane``

See :doc:`validation` for the distinction between implemented behavior and behavior covered by regression tests.
