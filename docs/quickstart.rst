Quickstart
==========

The examples below use the current supported high-level and low-level API surface.

High-Level Scalar Calls
-----------------------

.. code-block:: python

   import chillprop.highlevel as CH

   rho = CH.PropsSI("D", "T", 300.0, "P", 1e6, "Air")
   h = CH.PropsSI("H", "T", 300.0, "P", 1e6, "Nitrogen")
   phase = CH.PhaseSI("T", 300.0, "P", 1e6, "Nitrogen")

   print(rho, h, phase)

Batched JAX Calls
-----------------

.. code-block:: python

   import jax.numpy as jnp
   import chillprop.highlevel as CH

   T = jnp.array([280.0, 300.0, 320.0])
   P = jnp.array([5e5, 1e6, 2e6])

   h = CH.PropsSI("H", "T", T, "P", P, "Nitrogen")
   print(h)

Autodiff Example
----------------

.. code-block:: python

   import jax
   import chillprop.highlevel as CH

   dh_dP = jax.grad(
       lambda p: CH.PropsSI("H", "T", 400.0, "P", p, "Nitrogen")
   )(5e6)

   print(dh_dP)

AbstractState Example
---------------------

.. code-block:: python

   import chillprop.highlevel as CH

   state = CH.AbstractState("HEOS", "Nitrogen")
   state.update(CH.PT_INPUTS, 1e6, 300.0)

   print(state.rhomolar())
   print(state.hmass())
   print(state.viscosity())

Where To Go Next
----------------

* :doc:`examples` expands the quickstart snippets into task-oriented usage patterns.
* :doc:`high_level_api` documents supported ``PropsSI``-style behavior.
* :doc:`low_level_api` documents ``AbstractState`` and input-pair constants.
