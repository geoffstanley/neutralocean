Usage
=====

Installation
------------

The following is the easy installation that we aspire to.  This won't work, yet...

Install using either pip,

.. code-block:: console

    $ pip install neutralocean

or Conda,

.. code-block:: console

	$ conda install neutralocean -c conda-forge



Calculating approximately neutral surfaces
------------------------------------------

potential_surf
~~~~~~~~~~~~~~
.. autofunction:: neutralocean.surface.trad.potential_surf

anomaly_surf
~~~~~~~~~~~~
.. autofunction:: neutralocean.surface.trad.anomaly_surf

omega_surf
~~~~~~~~~~
.. autofunction:: neutralocean.surface.omega.omega_surf


Calculating other neutral things
--------------------------------

ntp_bottle_to_cast
~~~~~~~~~~~~~~~~~~
.. autofunction:: neutralocean.traj.ntp_bottle_to_cast

neutral_trajectory
~~~~~~~~~~~~~~~~~~
.. autofunction:: neutralocean.traj.neutral_trajectory

veronis_density
~~~~~~~~~~~~~~~
.. autofunction:: neutralocean.label.veronis_density

Choosing the Equation Of State
------------------------------

make_eos
~~~~~~~~
.. autofunction:: neutralocean.eos.tools.make_eos

vectorize_eos
~~~~~~~~~~~~~
.. autofunction:: neutralocean.eos.tools.vectorize_eos

make_eos_bsq
~~~~~~~~~~~~~
.. autofunction:: neutralocean.eos.tools.make_eos_bsq


JMD95 Equation of State
-----------------------
.. automodule:: neutralocean.eos.densjmd95

rho
~~~
.. autofunction:: neutralocean.eos.densjmd95.rho

rho_s_t
~~~~~~~
.. autofunction:: neutralocean.eos.densjmd95.rho_s_t

rho_p
~~~~~
.. autofunction:: neutralocean.eos.densjmd95.rho_p

TEOS-10 GSW Equation of State
-----------------------------

rho
---
.. autofunction:: neutralocean.eos.gsw.rho

rho_s_t
~~~~~~~
.. autofunction:: neutralocean.eos.gsw.rho_s_t

rho_p
~~~~~
.. autofunction:: neutralocean.eos.gsw.rho_p