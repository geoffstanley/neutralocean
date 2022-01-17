# neutralocean
Calculate neutral surfaces in the ocean, using Python

# Installation
Packaging this project for pip and conda is a work currently in progress.  At this stage, there is a pip version on TestPyPi.  Since TestPyPi doesn't provide dependencies, you'll have to manually pre-install a few packages before installing `neutralocean`.  The following instructions assume you are using conda to manage your environment.  Probably something similar works for venv?  

1.  Make and switch to a new environment (here named `nocean`, for example):
```
$ conda create --name nocean
$ conda activate nocean
```

2. Install dependencies.  First we install `numpy` and `numba` so that a somewhat older version of `python` will be installed: it seems `numba` does not work with `python3.10` ... and numpy 1.22 does not work with numba.vectorize right now.  How annoying.  
Also, we'll install everything from `conda-forge` since that's where we must get `gsw`, and I find it's better (in terms of resolving the environment) to have everything or nothing from `conda-forge`.
```
$ conda install -c conda-forge numpy==1.21 numba
$ conda install -c conda-forge scipy scikit-sparse xarray gsw
```
That does it for dependencies.  You can also install some other obviously useful things here (or later), like `ipython`, `matplotlib`, `pytest`, etc.
To install `neutralocean` from TestPyPi, we need `pip`, so do
```
$ conda install -c conda-forge pip
```

3. Install `neutralocean`:
```
$ pip install -i https://test.pypi.org/simple/ neutralocean==0.0.7
```

4. Test it out:
Try running the example script, `run_OCCA.py`.
```
$ python /path/to/neutralocean/neutralocean/examples/run_OCCA.py
```
