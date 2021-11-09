Usage
=====

Installation
------------

Install using either pip,

.. code-block:: console

    $ pip install neutral_surfaces

or Conda,

.. code-block:: console

	$ conda install neutral_surfaces -c conda-forge



Calculating approximately neutral surfaces
------------------------------------------

sigma_surf
~~~~~~~~~~
.. autofunction:: neutral_surfaces.sigma_surf

delta_surf
~~~~~~~~~~
.. autofunction:: neutral_surfaces.delta_surf

omega_surf
~~~~~~~~~~
.. autofunction:: neutral_surfaces.omega_surf


Calculating other neutral things
--------------------------------

veronis_density
~~~~~~~~~~~~~~~
.. autofunction:: neutral_surfaces.ntp.veronis_density

ntp_bottle_to_cast
~~~~~~~~~~~~~~~~~~
.. autofunction:: neutral_surfaces.ntp.ntp_bottle_to_cast

neutral_trajectory
~~~~~~~~~~~~~~~~~~
.. autofunction:: neutral_surfaces.ntp.neutral_trajectory


Choosing the Equation Of State
------------------------------

make_eos
~~~~~~~~
.. autofunction:: neutral_surfaces.eostools.make_eos