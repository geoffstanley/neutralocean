# neutralocean
Calculate neutral surfaces in the ocean, using Python

# Installation
If you use pip, simply execute
```
$ pip install neutralocean
```

If you use conda, we're going to still use pip because I haven't uploaded `neutralocean` to conda-forge yet.  I will do that soon.  In the meantime, it's probably best to install the main dependencies from conda-forge, first.  
```
$ conda install -c conda-forge numpy=1.21 numba
$ conda install -c conda-forge scipy scikit-sparse xarray gsw pip pooch
```
The use of `numpy=1.21` is currently necessary for `numba`.
Now that we've got most things from conda-forge, we'll get just `neutralocean` from pip:
```
$ pip install neutralocean
```


# Usage
Try running the example script:
```
python /path/to/neutralocean/neutralocean/examples/run_OCCA.py
```
If that runs, we're in business.  
Take a look at that example script, `run_OCCA.py` located in `neutralocean/examples/`.
It is documented and will walk you through loading an ocean model dataset, selecting an equation of state, calculating various approximately neutral surfaces, and more. 