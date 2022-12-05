Examples
********

Here, we apply ``neutralocean`` to two types of ocean model, `OCCA and ECCOv4 <https://www.ecco-group.org/products.htm>`_.

OCCA
====

`OCCA <https://doi.org/10.1175/2009JPO4043.1>`_ is an ocean atlas on a latitude-longitude grid.  The ``run_OCCA.py`` example script (below, and on `github <https://github.com/geoffstanley/neutralocean/blob/main/neutralocean/examples/run_OCCA.py>`_) shows more of ``neutralocean``'s capabilities.  It downloads the dataset, loads the data into memory, selects an equation of state, calculates various approximately neutral surfaces, and more.   

.. literalinclude:: ../neutralocean/examples/run_OCCA.py

ECCOv4
======

`ECCOv4 <https://www.ecco-group.org/products-ECCO-V4r4.htm>`_ uses a "lat-lon-cap" tiled rectilinear grid.  Whereas the OCCA script downloaded the dataset for you via FTP, for ECCOv4r4 you must register for a free Earthdata Login then download the dataset manually.  Follow the instructions in the ``print`` lines in the ``run_ECCOv4r4.py`` script (below and on `github <https://github.com/geoffstanley/neutralocean/blob/main/neutralocean/examples/run_ECCOv4r4.py>`_).

.. literalinclude:: ../neutralocean/examples/run_ECCOv4r4.py

