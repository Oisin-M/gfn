gfn
===

**gfn** is a Python library implementing Graph Feedforward Networks (GFNs) :cite:`gfn`, a generalisation of standard feedforward networks for graphical data.

.. image:: ../images/gfn.png
   :alt: A schematic of a GFN-based autoencoder.
   :align: center

Key advantages of GFNs
----------------------

- Resolution invariance
- Equivalence to feedforward networks for single fidelity data (no deterioration in performance)
- Provable guarantees on performance for super- and sub-resolution
- Both fixed and adapative multifidelity training possible

Installation
------------

You can install **gfn** via pip:

.. code-block:: bash

   pip install gfn-layer

**Note:** the package name on PyPI is **gfn-layer**.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   motivation
   userguide
   api
   references
   Paper <https://doi.org/10.1016/j.cma.2024.117458>
