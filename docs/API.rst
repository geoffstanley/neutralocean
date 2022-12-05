Public Function Reference
*************************

.. autosummary::
   :toctree: generated

Approximately Neutral Surfaces
==============================

.. _potdens_surf:

Potential Density Surface
-------------------------
.. autofunction:: neutralocean.surface.trad.potential_surf

.. _sva_surf:

Specific Volume Anomaly Surface
-------------------------------
.. autofunction:: neutralocean.surface.trad.anomaly_surf

.. _omega_surf:

Omega Surface
-------------
.. autofunction:: neutralocean.surface.omega.omega_surf

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
.. autofunction:: neutralocean.label.veronis_density

Mixed Layer
===========
.. autofunction:: neutralocean.mixed_layer.mixed_layer

Equation Of State
=================

Tools 
-----
.. autofunction:: neutralocean.eos.tools.make_eos

.. autofunction:: neutralocean.eos.tools.make_eos_s_t

.. autofunction:: neutralocean.eos.tools.make_eos_p

.. autofunction:: neutralocean.eos.tools.vectorize_eos

JMD95
-----
.. automodule:: neutralocean.eos.jmd95

.. autofunction:: neutralocean.eos.jmd95.rho

.. autofunction:: neutralocean.eos.jmd95.rho_s_t

.. autofunction:: neutralocean.eos.jmd95.rho_p

JMDFWG06
--------
.. automodule:: neutralocean.eos.jmdfwg06

.. autofunction:: neutralocean.eos.jmdfwg06.rho

.. autofunction:: neutralocean.eos.jmdfwg06.rho_s_t

.. autofunction:: neutralocean.eos.jmdfwg06.rho_p

TEOS-10 GSW
-----------
.. automodule:: neutralocean.eos.gsw

.. autofunction:: neutralocean.eos.gsw.specvol

.. autofunction:: neutralocean.eos.gsw.specvol_s_t

.. autofunction:: neutralocean.eos.gsw.specvol_p

(Vertical) Interpolation
========================

On the Fly
----------
.. automodule:: neutralocean.interp1d.__init__

.. autofunction:: neutralocean.interp1d.make_interpolator


Piecewise Polynomial (PP) coefficients
--------------------------------------
.. automodule:: neutralocean.ppinterp.__init__

.. autofunction:: neutralocean.ppinterp.linear.linear_coeffs

.. autofunction:: neutralocean.ppinterp.pchip.pchip_coeffs

.. autofunction:: neutralocean.ppinterp.ppval

.. autofunction:: neutralocean.ppinterp.ppval_two

Synthetic Ocean
===============
.. autofunction:: neutralocean.synthocean.synthocean