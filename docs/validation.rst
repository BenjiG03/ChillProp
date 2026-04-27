Validation
==========

This page summarizes the current automated parity coverage exercised against CoolProp for the documented supported materials. The statistics below are derived from the same grids used by the regression suite in ``tests/test_highlevel_pure_jax.py``.

Validation Scope
----------------

* Trivial fluid constants: 11 checks per fluid.
* Single-phase core parity grid: 4 state points by 20 outputs per fluid.
* Transport parity grid: 4 state points by 3 outputs for ``Argon``, ``Hydrogen``, ``Nitrogen``, ``Oxygen``, and ``Propane``.
* Two-phase parity grid: 6 state points by 10 outputs for fluids that CoolProp reports as pure.

Tolerances
----------

+----------------------------+-------------------+--------------------------------------------------------------+
| Category                   | Default tolerance | Notes                                                        |
+============================+===================+==============================================================+
| Trivial outputs            | ``3e-2``          | Scalar constants such as critical properties and molar mass  |
+----------------------------+-------------------+--------------------------------------------------------------+
| Single-phase core outputs  | ``5e-8``          | Property-specific relaxations are applied in the test suite  |
+----------------------------+-------------------+--------------------------------------------------------------+
| Two-phase outputs          | ``1e-3``          | ``Q`` uses ``1e-4``                                          |
+----------------------------+-------------------+--------------------------------------------------------------+
| Transport outputs          | ``5e-3``          | Viscosity, conductivity, and Prandtl number                  |
+----------------------------+-------------------+--------------------------------------------------------------+

Supported Materials Summary
---------------------------

+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| Fluid            | Trivial max rel err    | Single-phase max rel err   | Transport max rel err      | Two-phase max rel err      | Notes                                                         |
+==================+========================+============================+============================+============================+===============================================================+
| ``Air``          | ``0.000e+00``          | ``2.357e-08``              | ``n/a``                    | ``n/a``                    | No automated transport or two-phase grid in the current suite |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Argon``        | ``1.120e-07``          | ``7.508e-09``              | ``4.493e-08``              | ``5.121e-13``              | Full documented suite                                         |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``CarbonDioxide``| ``2.205e-07``          | ``2.871e-11``              | ``n/a``                    | ``3.574e-13``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Ethane``       | ``4.560e-09``          | ``7.822e-09``              | ``n/a``                    | ``4.433e-11``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Hydrogen``     | ``3.946e-04``          | ``1.276e-10``              | ``7.179e-06``              | ``5.855e-14``              | Full documented suite                                         |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``IsoButane``    | ``1.716e-05``          | ``7.918e-11``              | ``n/a``                    | ``5.168e-04``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Methane``      | ``9.522e-07``          | ``1.720e-09``              | ``n/a``                    | ``3.701e-13``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``n-Butane``     | ``4.588e-09``          | ``2.890e-09``              | ``n/a``                    | ``5.696e-12``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``n-Dodecane``   | ``3.134e-04``          | ``1.604e-10``              | ``n/a``                    | ``3.231e-11``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Nitrogen``     | ``1.309e-07``          | ``3.271e-09``              | ``3.780e-08``              | ``1.116e-12``              | Full documented suite                                         |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Oxygen``       | ``2.157e-02``          | ``4.389e-09``              | ``3.115e-08``              | ``3.401e-12``              | Full documented suite                                         |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Propane``      | ``8.156e-06``          | ``4.451e-12``              | ``6.556e-04``              | ``2.847e-10``              | Full documented suite                                         |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+
| ``Water``        | ``2.865e-09``          | ``5.327e-11``              | ``n/a``                    | ``3.974e-11``              | No automated transport grid                                   |
+------------------+------------------------+----------------------------+----------------------------+----------------------------+---------------------------------------------------------------+

Interpretation
--------------

* The parity plots below summarize ``D``, ``H``, ``S``, and ``A`` over the automated single-phase state grid.
* Transport properties are implemented for more fluids than the automated transport parity grid currently covers.
* Two-phase statistics are reported only where the automated saturation grid applies.

Parity Plots
------------

Air
^^^

.. image:: plots/validated/Air_parity.png
   :alt: Air parity plot

Argon
^^^^^

.. image:: plots/validated/Argon_parity.png
   :alt: Argon parity plot

CarbonDioxide
^^^^^^^^^^^^^

.. image:: plots/validated/CarbonDioxide_parity.png
   :alt: CarbonDioxide parity plot

Ethane
^^^^^^

.. image:: plots/validated/Ethane_parity.png
   :alt: Ethane parity plot

Hydrogen
^^^^^^^^

.. image:: plots/validated/Hydrogen_parity.png
   :alt: Hydrogen parity plot

IsoButane
^^^^^^^^^

.. image:: plots/validated/IsoButane_parity.png
   :alt: IsoButane parity plot

Methane
^^^^^^^

.. image:: plots/validated/Methane_parity.png
   :alt: Methane parity plot

n-Butane
^^^^^^^^

.. image:: plots/validated/n-Butane_parity.png
   :alt: n-Butane parity plot

n-Dodecane
^^^^^^^^^^

.. image:: plots/validated/n-Dodecane_parity.png
   :alt: n-Dodecane parity plot

Nitrogen
^^^^^^^^

.. image:: plots/validated/Nitrogen_parity.png
   :alt: Nitrogen parity plot

Oxygen
^^^^^^

.. image:: plots/validated/Oxygen_parity.png
   :alt: Oxygen parity plot

Propane
^^^^^^^

.. image:: plots/validated/Propane_parity.png
   :alt: Propane parity plot

Water
^^^^^

.. image:: plots/validated/Water_parity.png
   :alt: Water parity plot
