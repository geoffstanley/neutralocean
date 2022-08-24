Usage
=====

Installation
------------

Install using pip,

.. code-block:: console

    $ pip install neutralocean

At the moment, it seems numba does not work with python3.10, and numpy 1.22 does not work with numba.vectorize right now. How annoying.  So if the above doesn't work, try manually installing the dependencies as follows.  First install numpy 1.21 to get a slightly older version of Python, then pull in the others::

	$ conda install -c conda-forge numpy==1.21 numba
	$ conda install -c conda-forge xarray scipy scikit-sparse gsw pooch

We're installing everything from conda-forge since that's where we must get `gsw`, and I find it's better (in terms of resolving the environment) to have everything or nothing from conda-forge.


Getting Started
---------------
Have a look at the script in `run_OCCA.py` in `neutralocean/examples/`
