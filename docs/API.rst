Public Function Reference
*************************

.. autosummary::
   :toctree: generated

Approximately Neutral Surfaces
==============================

.. _potdens_surf:

Potential Density Surface
-------------------------
.. autofunction:: neutralocean.surface.potential_surf

.. _sva_surf:

Specific Volume Anomaly Surface
-------------------------------
.. autofunction:: neutralocean.surface.anomaly_surf

.. _omega_surf:

Omega Surface
-------------
.. autofunction:: neutralocean.surface.omega_surf

Grids
=====

Rectilinear
-----------
.. autofunction:: neutralocean.grid.rectilinear.build_grid

.. autofunction:: neutralocean.grid.rectilinear.edgedata_to_maps

xgcm (tiled rectilinear)
------------------------
.. autofunction:: neutralocean.grid.xgcm.build_grid

.. autofunction:: neutralocean.grid.xgcm.edgedata_to_maps

tripolar
--------

.. autofunction:: neutralocean.grid.tripolar.build_grid

.. autofunction:: neutralocean.grid.tripolar.edgedata_to_maps

Graph
-----
.. autofunction:: neutralocean.grid.graph.build_grid

.. autofunction:: neutralocean.grid.graph.edges_to_graph

.. autofunction:: neutralocean.grid.graph.graph_binary_fcn

Neutrality Error
================
.. autofunction:: neutralocean.ntp.ntp_epsilon_errors

.. autofunction:: neutralocean.ntp.ntp_epsilon_errors_norms

Neutral Trajectory
==================
.. autofunction:: neutralocean.traj.ntp_bottle_to_cast

.. autofunction:: neutralocean.traj.neutral_trajectory

Veronis Density
===============
.. autofunction:: neutralocean.label.veronis

Mixed Layer
===========
.. autofunction:: neutralocean.mixed_layer.mld

Static Stability
================
.. autofunction:: neutralocean.stability.count_unstable

.. autofunction:: neutralocean.stability.stabilize_ST

Equation Of State
=================

Tools 
-----
.. autofunction:: neutralocean.eos.tools.load_eos

.. autofunction:: neutralocean.eos.tools.make_bsq

.. autofunction:: neutralocean.eos.tools.vectorize_eos



(Vertical) Interpolation using Piecewise Polynomials (PP)
=========================================================

.. automodule:: neutralocean.ppinterp.__init__

.. autofunction:: neutralocean.ppinterp.tools.make_pp

.. autofunction:: neutralocean.ppinterp.ppval

.. autofunction:: neutralocean.ppinterp.ppval_1

.. autofunction:: neutralocean.ppinterp.ppval_two

.. autofunction:: neutralocean.ppinterp.ppval_1_two


Data
====

.. autofunction:: neutralocean.data.load_OCCA

.. autofunction:: neutralocean.data.synthocean