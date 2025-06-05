Examples
********

Here, we apply ``neutralocean`` to various ocean models using different horizontal grid structures.

.. _ex OCCA:

OCCA (latitude-longitude)
=========================

`OCCA <https://doi.org/10.1175/2009JPO4043.1>`_ is an ocean atlas on a latitude-longitude grid. The ``run_OCCA.py`` example script (below, and on `github <https://github.com/geoffstanley/neutralocean/blob/main/examples/run_OCCA.py>`_) shows more of ``neutralocean``'s capabilities. It downloads the dataset, loads the data into memory, selects an equation of state, calculates various approximately neutral surfaces, and more.   

.. literalinclude:: ../examples/run_OCCA.py

.. _ex ECCOv4:

ECCOv4 (lat-lon-cap)
====================

`ECCOv4 <https://www.ecco-group.org/products-ECCO-V4r4.htm>`_ uses a "lat-lon-cap" tiled rectilinear grid. Whereas the OCCA script downloaded the dataset for you via FTP, for ECCOv4r4 you must register for a free Earthdata Login then download the dataset manually. Follow the instructions in the ``print`` lines in the ``run_ECCOv4r4.py`` script (below and on `github <https://github.com/geoffstanley/neutralocean/blob/main/examples/run_ECCOv4r4.py>`_).

.. literalinclude:: ../examples/run_ECCOv4r4.py

.. _ex ORCA:

ORCA (tripolar)
===============

The `ORCA <https://www.nemo-ocean.eu/doc/node108.html>`_ tripolar grid used by `NEMO <https://www.nemo-ocean.eu/>`_ is similar to a latitude-longitude grid, but it splits the north pole singularity into two pieces and places these over land. The ``run_NEMO_ORCA1.py`` example (below and on `github <https://github.com/geoffstanley/neutralocean/blob/main/examples/run_NEMO_ORCA1.py>`_) applies ``neutralocean`` to `CanESM5 <https://gmd.copernicus.org/articles/12/4823/2019/gmd-12-4823-2019.html>`_ data in `CMIP6 <https://esgf-node.llnl.gov/projects/cmip6/>`_.

.. literalinclude:: ../examples/run_NEMO_ORCA1.py