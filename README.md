# neutralocean
Calculate neutral surfaces in the ocean, using Python

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

Uploading the sphinx-generated documentation to a website remains a work-in-progress.  Sorry.  Get in touch with me if you want to help, or need help.  In the meantime, read the documentation inside the code -- it's essentially the same.