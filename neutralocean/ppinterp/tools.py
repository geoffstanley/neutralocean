import functools as ft
from importlib import import_module


@ft.lru_cache(maxsize=10)
def make_pp(
    interpolant="linear",
    kind="u",
    out="coeffs",
    nans=True,
    num_dep_vars=1,
):
    """Make function for piecewise polynomial interpolation.
    
    Parameters
    ----------
    
    interpolant : str, Default "linear"
        Name of the interpolant to use. Currently, either "linear" or "pchip".
        
    kind : str, Default "u"
        Either "1" (single; make a function that performs one interpolation) 
        or "u" (universal; make a function performs a low level loop over multiple 
        interpolation problems, using `numba`'s `guvectorize` decorator). 
        
        If "1", the returned function `f` operates on independent and dependent data, `X` 
        and `Y`, that are 1D arrays, and (if `out="interp"`) the evaluation site `x` is a scalar. 
        
        If "u", the returned function `f` operates on independent and dependent data, `X`
        and `Y`, may be N+1 dimensional arrays that are mutually broadcastable, 
        and (if `out="interp"`) the evaluation site `x` may be a N dimensional array that 
        is broadcastable to the first N dimensions of `X` or `Y`.
        
    out : str, Default "coeffs"
        Specifies the type of output returned by `f`.
        
        If `"coeffs"`, then `f` returns the piecewise polynomial coefficients, and `f` 
        does not take in the evaluation sites `x`.
        
        If `"interp"`, then `f` evaluates the interpolant(s) at the evaluation site `x`. 
    
    nans : bool, Default True
        If True, the independent and dependent data, `X` and `Y`, may have NaNs.
        
        If False, `X` and `Y`, must have no NaNs. In this case, a slightly faster function
        is created that skips the step of checking for the first contiguous block of 
        non-NaN data (performed by `valid_range_1_two`).
    
    num_dep_vars : int, Default 1
        If 2, `f` takes two dependent variables (`Y` and `Z`) that are both evaluated at
        the evaluation site `x`. This is slightly faster than calling an interpolation 
        function twice, because the location of the evaluation site `x` within the 
        dependent data `X` need only be calculated once.
        Note if `out = "coeffs"` then `num_dep_vars = 1` is required. In that case, create
        two pairs of piecewise polynomail coefficients `Yppc` and `Zppc` then evaluate 
        them at `x` using `ppval_two`. 
        
    Returns
    -------
    f : function
        Interpolating function. The possible inputs to `f` are 
        
        - `x` the evaluation site(s), (only if `out = "interp"`)
        - `X` the independent data
        - `Y` the dependent data
        - `Z` additional dependent data (only if `num_dep_vars = 2`)
            
        The outputs of `f` are
        
        - `Yppc` the piecewise polynomial coefficients (if `out = "coeffs"`) OR
        - `y` the interpolant evaluated at `x` (if `out = "interp"), OR
        - `(y, z)` the interpolants evaluated at `x` (if `out = "interp" and `num_dep_vars=2`).
            
    """

    if interpolant not in ("linear", "pchip"):
        raise ValueError(
            f"Expected `interpolant` in ('lienar', 'pchip'); got {interpolant}"
        )

    if kind not in ("1", "u"):
        raise ValueError(f"Expected `kind` in ('1', 'u'); got {kind}")

    if out not in ("coeffs", "interp"):
        raise ValueError(f"Expected `out` in ('coeffs', 'interp'); got {out}")

    if num_dep_vars not in (1, 2):
        raise ValueError("Expected `num_dep_vars` in (1, 2); got {num_dep_vars}")

    if out == "coeffs" and num_dep_vars != 1:
        raise ValueError("With `out='coeffs'`, currently only handles `num_dep_vars=1`")

    if nans == False:
        if out == "interp" or kind == "u":
            raise ValueError("With `nans=False`, expected `out='coeffs'` and `kind='1'`.")

    # Begin programmatically generating the function name to import
    if kind == "1":
        kind = "_1"
    elif kind == "u":
        kind = ""

    if nans == False and interpolant == "pchip":
        nans = "_nonan"
    else:
        # if interpolant == "linear", just use regular coeffs_1 function.
        nans = ""

    if num_dep_vars == 1:
        num_dep_vars = ""
    elif num_dep_vars == 2:
        num_dep_vars = "_two"

    fcn_name = interpolant + "_" + out + kind + nans + num_dep_vars

    # Below is equivalent to
    # from neutralocean.ppinterp.`interpolant` import `fcn_name`
    return import_module("neutralocean.ppinterp." + interpolant).__getattribute__(
        fcn_name
    )