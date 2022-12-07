# neutralocean

<a href="https://neutralocean.readthedocs.org">
    <img alt="latest docs" src="https://img.shields.io/badge/documentation-latest-blue">
</a>

<a href="https://badge.fury.io/py/neutralocean">
    <img alt="latest docs" src="https://badge.fury.io/py/neutralocean.svg">
</a>

<a href="https://anaconda.org/conda-forge/neutralocean">
    <img alt="latest docs" src="https://img.shields.io/conda/vn/conda-forge/neutralocean.svg">
</a>

Calculate neutral surfaces in the ocean, using Python.

The major task of this software is to calculate **omega-surfaces**, following the algorithm of [Stanley et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436).  Omega surfaces are highly accurate approximately neutral surfaces that work by iteratively reducing the neutrality error. 

There are also routines to calculate potential density surfaces, specific volume anomaly surfaces, neutral trajectories, the Veronis density label, and measures of neutrality error.

**Topobaric surfaces** [(Stanley, 2019a)](https://www.sciencedirect.com/science/article/pii/S1463500318302221) in their modified form [(Stanley 2019b)](https://www.sciencedirect.com/science/article/pii/S1463500318302233) are the most accurate approximately neutral surfaces that posses an exact geostrophic streamfunction (furnishing an Ertel potential vorticity with no baroclinic production term).  Software to compute topobaric surfaces is significantly more complicated and is currently only available in the original MATLAB [neutral-surfaces](https://github.com/geoffstanley/neutral-surfaces) toolbox.

**How to cite?** If you use this software, the most appropriate paper to cite is [Stanley et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436).

# Installation
Simply execute either
```
$ pip install neutralocean
```
if you use pip, or
```
$ conda install -c conda-forge neutralocean
```
if you use conda.

# Documentation
See <https://neutralocean.readthedocs.org>
