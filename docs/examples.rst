Examples
========

This page collects representative usage patterns supported by the current implementation.

Scalar ``PropsSI`` Evaluation
-----------------------------

.. code-block:: python

   import chillprop.highlevel as CH

   density = CH.PropsSI("D", "T", 298.15, "P", 1.0e6, "CarbonDioxide")
   entropy = CH.PropsSI("S", "T", 298.15, "P", 1.0e6, "CarbonDioxide")

Mass-Basis and Molar-Basis Outputs
----------------------------------

The high-level interface accepts CoolProp-style aliases for mass-basis and molar-basis properties.

.. code-block:: python

   import chillprop.highlevel as CH

   h_mass = CH.PropsSI("H", "T", 300.0, "P", 1e6, "Water")
   h_molar = CH.PropsSI("Hmolar", "T", 300.0, "P", 1e6, "Water")

Batched Evaluation
------------------

When either input value is array-like, ChillProp broadcasts the inputs and evaluates the requested property over the resulting grid.

.. code-block:: python

   import jax.numpy as jnp
   import chillprop.highlevel as CH

   temperatures = jnp.array([280.0, 290.0, 300.0, 310.0])
   pressures = jnp.array([2e5, 5e5, 1e6, 2e6])

   cp = CH.PropsSI("C", "T", temperatures, "P", pressures, "Argon")

Gradients
---------

The high-level scalar interface can participate in JAX differentiation for supported state specifications.

.. code-block:: python

   import jax
   import chillprop.highlevel as CH

   dmu_dT = jax.grad(
       lambda temp: CH.PropsSI("V", "T", temp, "P", 5e5, "Nitrogen")
   )(300.0)

``AbstractState`` Workflow
--------------------------

``AbstractState`` is useful when several outputs are needed for the same updated state.

.. code-block:: python

   import chillprop.highlevel as CH

   state = CH.AbstractState("HEOS", "Propane")
   state.update(CH.DmolarT_INPUTS, 5000.0, 320.0)

   result = {
       "pressure": state.p(),
       "enthalpy_molar": state.hmolar(),
       "entropy_molar": state.smolar(),
       "conductivity": state.conductivity(),
   }

Phase and Saturation Usage
--------------------------

Two-phase usage is currently limited to the implemented input/output subset. Supported examples include ``T``/``Q`` state specifications and phase identification.

.. code-block:: python

   import chillprop.highlevel as CH

   h_vapor = CH.PropsSI("H", "T", 120.0, "Q", 1.0, "Nitrogen")
   phase = CH.PhaseSI("T", 120.0, "Q", 1.0, "Nitrogen")

Additional Material
-------------------

The repository also includes a notebook at ``docs/notebooks/Usage_Guide.ipynb`` for exploratory usage outside the formal documentation build.
