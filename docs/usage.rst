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



Calculating Neutral Surfaces
----------------------------

sigma_surf
~~~~~~~~~~
.. autofunction:: neutral_surfaces.sigma_surf
.. autofunction:: neutral_surfaces.delta_surf
.. autofunction:: neutral_surfaces.omega_surf


Calculating Other Neutral Things
--------------------------------

veronis_density
~~~~~~~~~~~~~~~
.. autofunction:: neutral_surfaces.lib.veronis_density

Choosing the Equation of State
------------------------------

make_eos
~~~~~~~~
.. autofunction:: neutral_surfaces.eostools.make_eos