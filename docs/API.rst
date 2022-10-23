Public Function Reference
=========================

.. autosummary::
   :toctree: generated

Potential density surfaces
--------------------------
.. autofunction:: neutralocean.surface.trad.potential_surf

In-situ density / specific volume anomaly surfaces
--------------------------------------------------
.. autofunction:: neutralocean.surface.trad.anomaly_surf

Omega surfaces
--------------
.. autofunction:: neutralocean.surface.omega.omega_surf

Neutrality Error
----------------
.. autofunction:: neutralocean.ntp.ntp_epsilon_errors

.. autofunction:: neutralocean.ntp.ntp_epsilon_errors_norms

Neutral Tangent Plane - Bottle to Cast
--------------------------------------
.. autofunction:: neutralocean.traj.ntp_bottle_to_cast

Neutral Trajectory
------------------
.. autofunction:: neutralocean.traj.neutral_trajectory

Veronis Density
---------------
.. autofunction:: neutralocean.label.veronis_density

Choosing the Equation Of State
------------------------------

make_eos
~~~~~~~~
.. autofunction:: neutralocean.eos.tools.make_eos

vectorize_eos
~~~~~~~~~~~~~
.. autofunction:: neutralocean.eos.tools.vectorize_eos

make_bsq
~~~~~~~~~~~~~
.. autofunction:: neutralocean.eos.tools.make_bsq


JMD95 Equation of State
-----------------------
.. automodule:: neutralocean.eos.jmd95

rho
~~~
.. autofunction:: neutralocean.eos.jmd95.rho

rho_s_t
~~~~~~~
.. autofunction:: neutralocean.eos.jmd95.rho_s_t

rho_p
~~~~~
.. autofunction:: neutralocean.eos.jmd95.rho_p

JMDFWG06 Equation of State
--------------------------
.. automodule:: neutralocean.eos.jmdfwg06

rho
~~~
.. autofunction:: neutralocean.eos.jmdfwg06.rho

rho_s_t
~~~~~~~
.. autofunction:: neutralocean.eos.jmdfwg06.rho_s_t

rho_p
~~~~~
.. autofunction:: neutralocean.eos.jmdfwg06.rho_p

TEOS-10 GSW Equation of State
-----------------------------

rho
~~~
.. autofunction:: neutralocean.eos.gsw.rho

rho_s_t
~~~~~~~
.. autofunction:: neutralocean.eos.gsw.rho_s_t

rho_p
~~~~~
.. autofunction:: neutralocean.eos.gsw.rho_p


Vertical Interpolation: on the fly
----------------------------------

make_interpolator
~~~~~~~~~~~~~~~~~
.. autofunction:: neutralocean.interp1d.make_interpolator


Vertical Interpolation: Piecewise Polynomial (PP) coefficients
--------------------------------------------------------------

linear_coeffs
~~~~~~~~~~~~~
.. autofunction:: neutralocean.ppinterp.linear.linear_coeffs

pchip_coeffs
~~~~~~~~~~~~
.. autofunction:: neutralocean.ppinterp.pchip.pchip_coeffs

ppval
~~~~~
.. autofunction:: neutralocean.ppinterp.ppval

ppval_two
~~~~~~~~~
.. autofunction:: neutralocean.ppinterp.ppval_two
