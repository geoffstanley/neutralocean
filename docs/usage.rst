Usage
=====

.. _installation:

Installation
------------

Simply execute either

.. code-block:: console

	(.venv) $ pip install neutralocean

if you use pip, or

.. code-block:: console

	(.venv) $ conda install -c conda-forge neutralocean

if you use conda.

Getting Started
---------------
Try running the example script (changing the initial path as needed):

.. code-block:: console

	python /path/to/neutralocean/neutralocean/examples/run_example_4casts.py

If you use miniconda, that might be in ``~/miniconda3/envs/<YOUR-ENVIRONMENT-NAME>/lib/python<VERSION-NUMBER>/site-packages/neutralocean/``.  You can also find this script `online <https://github.com/geoffstanley/neutralocean/blob/main/neutralocean/examples/run_example_4casts.py>`_.
If that runs, we're in business.  

Look at that example script, ``run_example_4casts.py``, to learn the basic usage of ``neutralocean``.

Then try running the ``run_OCCA.py`` script in the same folder.  Take a look at that code: it's documented and will walk you through loading an ocean model dataset, selecting an equation of state, calculating various approximately neutral surfaces, and more. 
