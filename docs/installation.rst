Getting Started
***************

.. _installation:

Installation
============

Simply execute either

.. code-block:: console

	(.venv) $ pip install neutralocean

if you use pip, or

.. code-block:: console

	(.venv) $ conda install -c conda-forge neutralocean

if you use conda.

.. _testexample:

Source Code
===========

The open-source `neutralocean` code is available at https://github.com/geoffstanley/neutralocean

Test Example
============

The example script `run_4casts.py` (copied below and on `github <https://github.com/geoffstanley/neutralocean/blob/main/examples/run_4casts.py>`_) creates synthetic salt, temperature, and pressure data for 4 vertical casts, specifies which pairs of these 4 casts are adjacent (see :ref:`grids:grids`), and then calculates a potential density surface, specific volume anomaly surface, and omega surface.  Epsilon neutrality errors are calculated for each link on the final omega surface.

Try running this script yourself, e.g. (changing path as needed)

.. code-block:: console

  python /path/to/neutralocean/examples/run_4casts.py

If that runs, your `neutralocean` package is working correctly.

Input
-----

.. literalinclude:: ../examples/run_4casts.py

Output
------

.. code-block:: console

	potential done |           4 wet casts | RMS(ϵ) = 2.95366114e-13  | 2.762 sec
	 ** The potential specific volume surface (referenced to 0.0dbar) with isovalue = 0.0009732360097323601 m3 kg-1 has root-mean-square ϵ neutrality error 2.953661135576716e-13 m2 kg-1.
	anomaly done |           4 wet casts | RMS(ϵ) = 1.21998943e-13  | 2.327 sec
	 ** The in-situ specific volume anomaly surface (referenced to (34.5, 4.0)) with isovalue = 0.0 m3 kg-1 has root-mean-square ϵ neutrality error 1.2199894286212603e-13 m2 kg-1.
	iter |    MAV(ϕ)     |    RMS(Δp)      | # wet casts (# new) |     RMS(ϵ)     | time (s)
	   0 |                                 |           4         | 7.05305721e-14 | 0.000
	   1 | 7.62146233e-09 | 1.77640515e+01 |           4 (    0) | 8.48888373e-15 | 3.417
	   2 | 6.02796769e-10 | 1.44982488e+00 |           4 (    0) | 6.12093753e-15 | 0.001
	   3 | 4.57931777e-11 | 1.11692359e-01 |           4 (    0) | 6.10611565e-15 | 0.001
	   4 | 3.38861795e-12 | 8.31849303e-03 |           4 (    0) | 6.10621916e-15 | 0.001
	   5 | 2.47457692e-13 | 6.09321842e-04 |           4 (    0) | 6.10623327e-15 | 0.001
	   6 | 1.79561251e-14 | 4.42768237e-05 |           4 (    0) | 6.10623432e-15 | 0.178
	 ** The omega-surface initialized from a potential density surface (referenced to 1500 dbar) intersecting the cast labelled '0' at pressure 1500 bar has root-mean-square ϵ neutrality error 6.1062343244329895e-15 m2 kg-1.
	The ϵ neutrality errors on the ω-surface are as follows:
	  From cast 0 to cast 1, ϵ = -6.946406152330754e-15 m^2 kg^-1
	  From cast 0 to cast 2, ϵ = 6.9464177373746585e-15 m^2 kg^-1
	  From cast 1 to cast 2, ϵ = -6.946399353678772e-15 m^2 kg^-1
	  From cast 2 to cast 3, ϵ = 2.4532753952971413e-20 m^2 kg^-1
	Note that the connection between casts 2 and 3 has virtually 0 neutrality error.  This is because cast 3 is ONLY connected to cast 2, so this link can be along the (discrete) neutral tangent plane joining cast 2 and 3. The ω-surface finds this.
