Installation
============

Requirements
------------

* Python 3.10 or newer.
* A working environment for ``jax`` and ``jaxlib`` compatible with the target platform.

Install From PyPI
-----------------

.. code-block:: bash

   pip install chillprop

Install For Local Development
-----------------------------

.. code-block:: bash

   git clone https://github.com/BenjiG03/ChillProp.git
   cd ChillProp
   pip install -e .

Documentation Build Dependencies
--------------------------------

The documentation site uses Sphinx and the Read the Docs theme.

.. code-block:: bash

   pip install -r docs/requirements.txt

Build The Documentation Locally
-------------------------------

From the repository root:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

The generated landing page is written to ``docs/_build/html/index.html``.
