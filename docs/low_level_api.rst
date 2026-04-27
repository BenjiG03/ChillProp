Low-Level API
=============

ChillProp exposes a compact ``AbstractState`` interface for workflows that benefit from updating a state once and retrieving several properties from that state.

Supported Construction
----------------------

Only the default backend and ``HEOS`` are supported.

.. code-block:: python

   import chillprop.highlevel as CH

   state = CH.AbstractState("HEOS", "Water")

Supported ``update()`` Input Pairs
----------------------------------

``AbstractState.update()`` currently supports:

* ``PT_INPUTS``
* ``HmassP_INPUTS``
* ``HmolarP_INPUTS``
* ``PSmass_INPUTS``
* ``PSmolar_INPUTS``
* ``QT_INPUTS``
* ``DmassT_INPUTS``
* ``DmolarT_INPUTS``

Property Accessors
------------------

The implemented scalar accessors include:

* density: ``rhomolar()``, ``rhomass()``
* temperature and pressure: ``T()``, ``p()``
* energy and entropy: ``hmolar()``, ``hmass()``, ``smolar()``, ``smass()``, ``umolar()``, ``umass()``
* heat capacities: ``cpmolar()``, ``cpmass()``, ``cvmolar()``, ``cvmass()``
* transport: ``viscosity()``, ``conductivity()``
* phase/saturation helpers: ``Q()``

Keyed Output
------------

``keyed_output()`` supports a subset of CoolProp keyed outputs covering:

* state variables
* thermodynamic properties
* transport properties
* phase index
* common trivial constants

Use generated API documentation in :doc:`api_reference` for the exact member list currently documented as public.

Current Limits
--------------

The low-level interface is intentionally narrower than CoolProp. Phase-envelope tools, mixture operations, the broader backend matrix, and the full mutable ``AbstractState`` surface are not implemented.
