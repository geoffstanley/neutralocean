# neutralocean
Calculate neutral surfaces in the ocean, using Python.

The major task of this software is to calculate omega-surfaces, following the algorithm of [Stanley et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436).  Omega surfaces are highly accurate approximately neutral surfaces that work by iteratively minimizing the neutrality error. 

There are also routines to calculate potential density surfaces, specific volume anomaly surfaces, neutral trajectories, the Veronis density label, and measures of neutrality error.

Topobaric surfaces [(Stanley, 2019a](https://www.sciencedirect.com/science/article/pii/S1463500318302221) in their modified form [(Stanley 2019b)](https://www.sciencedirect.com/science/article/pii/S1463500318302233) are the most accurate approximately neutral surfaces that posses an exact geostrophic streamfunction (furnishing an Ertel potential vorticity with no baroclinic production term).  Software to compute topobaric surfaces is significantly more complicated and is currently only available in the original MATLAB [neutral-surfaces](https://github.com/geoffstanley/neutral-surfaces) toolbox.

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

# Usage
Try running the example script (changing the initial path as needed):
```
python /path/to/neutralocean/neutralocean/examples/run_example_4casts.py
```
If you use miniconda, that might be in `~/miniconda3/envs/<YOUR-ENVIRONMENT-NAME>/lib/python<VERSION-NUMBER>/site-packages/neutralocean/`
If that runs, we're in business.  

Look at that example script, `run_example_4casts.py`, to learn the basic usage of `neutralocean`.

Then try running the `run_OCCA.py` script in the same folder.  Take a look at that code: it's documented and will walk you through loading an ocean model dataset, selecting an equation of state, calculating various approximately neutral surfaces, and more. 

# Documentation
Uploading the sphinx-generated documentation to a website remains a work-in-progress.  Sorry.  Get in touch with me if you want to help me, or need help with this code.  In the mean-time, the documentation inside the code is quite readable.