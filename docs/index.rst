.. Neutral surfaces documentation master file, created by
   sphinx-quickstart on Fri Apr  9 11:29:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Neutral Ocean 
=============

This Python package calculates approximately neutral surfaces (ANS) in the ocean.
The classic examples of ANSs are potential density surfaces and specific
volume anomaly surfaces.  The most accurate ANS, in terms of root-mean-square
epsilon neutrality errors, is the omega surface (Klocker et al 2009, Stanley et al 2021).
Another highly accurate and fast ANS that, unlike omega surfaces, comes with
an exact geostrophic streamfunction and hence Ertel potential vorticity,
is the topobaric surface (Stanley 2019a,b).  Topobaric surfaces are not yet
available in Python, however. 

The original MATLAB version (including topobaric surfaces) is available here:
https://github.com/geoffstanley/neutral-surfaces

This Sphinx auto generated documentation is a work in progress.  
You may or may not prefer to view the documentation in the code itself.


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

