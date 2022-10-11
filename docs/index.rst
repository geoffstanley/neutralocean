.. Neutral surfaces documentation master file, created by
   sphinx-quickstart on Fri Apr  9 11:29:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Neutral Ocean 
=============

Calculate neutral surfaces in the ocean, using Python.

The major task of this software is to calculate omega-surfaces, following the algorithm of `Stanley et al. (2021) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436>`_.  Omega surfaces are highly accurate approximately neutral surfaces that work by iteratively reducing the neutrality error. 

There are also routines to calculate potential density surfaces, specific volume anomaly surfaces, neutral trajectories, the Veronis density label, and measures of neutrality error.

Topobaric surfaces `(Stanley, 2019a) <https://www.sciencedirect.com/science/article/pii/S1463500318302221>`_ in their modified form `(Stanley 2019b) <https://www.sciencedirect.com/science/article/pii/S1463500318302233>`_ are the most accurate approximately neutral surfaces that posses an exact geostrophic streamfunction (furnishing an Ertel potential vorticity with no baroclinic production term).  Software to compute topobaric surfaces is significantly more complicated and is currently only available in the original MATLAB `neutral-surfaces <https://github.com/geoffstanley/neutral-surfaces>`_ toolbox.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   API
   internals

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========
Klocker, McDougall, Jackett 2009, A new method of forming approximately neutral surfaces, Ocean Science, 5, 155-172.

Stanley, G.J., 2019a. Neutral surface topology. Ocean Modelling 138, 88–106. https://doi.org/10.1016/j.ocemod.2019.01.008

Stanley, G.J., 2019b. The exact geostrophic streamfunction for neutral surfaces. Ocean Modelling 138, 107–121. https://doi.org/10.1016/j.ocemod.2019.04.002

Stanley, McDougall, Barker 2021, Algorithmic improvements to finding approximately neutral surfaces, Journal of Advances in Earth System Modelling, 13(5).

