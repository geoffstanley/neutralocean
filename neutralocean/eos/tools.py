"""Tools for handling the Equation of State"""

import functools as ft
import numpy as np
import numba as nb
import importlib
import inspect
import warnings

# Dictionary mapping names of modules in the same directory as this file to either
# "specvol" or "rho" depending on which variable they calculate.
modules = {"gsw": "specvol", "polyTEOS10bsq" : "rho", "jmd95": "rho", "jmdfwg06": "rho"}


@ft.lru_cache(maxsize=10)
def load_eos(eos, derivs="", grav=None, rho_c=None):
    """Load EOS function from library.

    Parameters
    ----------
    eos : str

        If a str, can be 
        - `'gsw'` to generate the 75 term approximation [1]_ of the TEOS-10 [2]_ specific volume,
        - `'polyTEOS10bsq'` to generate the Boussinesq polynomial approximation [1]_
        of the TEOS-10 in-situ density [2]_
        - `'jmd95'` to generate the Jackett and McDougall (1995) in-situ density [3]_, or 
        - `'jmdfwg06'` to generate the Jackett et al (2006) in-situ density [4]_.

    derivs : str, Default ""

        String specifying which partial derivatives of the EOS are desired.
        Only used when `eos` is a string.
        The actual function loaded is named `eos + derivs`.
        For example, "" loads the EOS itself, 
        "_p" will load the partial derivative with respect to p,
        "_s_t" will load the function whose two outputs are the s and t
        partial derivatives, respectively.

    grav, rho_c : float, Default None

        Gravitational acceleration [m s-2] and Boussinesq reference density
        [kg m-3]. If both are provided, the equation of state is modified as
        appropriate for the Boussinesq approximation, in which the third
        argument is depth, not pressure. Specifically, a depth `z` is
        converted to `1e-4 * grav * rho_c * z`, which is the hydrostatic
        pressure [dbar] at depth `z` [m] caused by a water column of density
        `rho_c` under gravity `grav`.

    Returns
    -------
    fn: function

        Equation of State function accepting three arguments: 
        (salinity, temperature, pressure) when `grav` or `rho_c` is None, or 
        (salinity, temperature, depth) otherwise.
    
    Notes
    -----
    .. [1] Roquet, F., G. Madec, Trevor J. McDougall, and Paul M. Barker. “Accurate
       Polynomial Expressions for the Density and Specific Volume of Seawater Using
       the TEOS-10 Standard.” Ocean Modelling 90 (June 2015): 29-43.
       https://doi.org/10.1016/j.ocemod.2015.04.002.
        
    .. [2] McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and
       the Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127,
       SBN 978-0-646-55621-5.

    .. [3] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388

    .. [4] Jackett, D. R., McDougall, T. J., Feistel, R., Wright, D. G., &
       Griffies, S. M. (2006). Algorithms for Density, Potential Temperature,
       Conservative Temperature, and the Freezing Temperature of Seawater.
       Journal of Atmospheric and Oceanic Technology, 23(12), 1709-1728.
       https://doi.org/10.1175/JTECH1946.1

    """
    
    if eos in modules:
        fcn_name = modules[eos] + derivs
        fn = importlib.import_module(
            "neutralocean.eos." + eos
        ).__getattribute__(fcn_name)
    else:
        raise ValueError(
            f"Equation of state {eos} not (yet) implemented."
            " Currently, eos must be one of " + modules.__str__()
        )

    if grav != None and rho_c != None:
        fn = make_bsq(fn, grav, rho_c)

    return fn


def make_eos(eos, grav=None, rho_c=None):
    """Make an equation of state function, possibly modified for Boussinesq

    Parameters
    ----------
    eos : str or function

        If a str, can be `'gsw'` to generate the TEOS-10 specific volume [1]_,
        `'jmd95'` to generate the Jackett and McDougall (1995) in-situ
        density [2]_, or `'jmdfwg06'` to generate the Jackett et al (2006)
        in-situ density [3]_.

        If a function, should be an equation of state as a function of
        practical / Absolute salinity, potential / Conservative temperature,
        and pressure (for non-Boussesinq) / depth (for Boussinesq).

    grav, rho_c : float

        Gravitational acceleration [m s-2] and Boussinesq reference density
        [kg m-3]. If both are provided, the equation of state is modified as
        appropriate for the Boussinesq approximation, in which the third
        argument is depth, not pressure. Specifically, a depth `z` is
        converted to `1e-4 * grav * rho_c * z`, which is the hydrostatic
        pressure [dbar] at depth `z` [m] caused by a water column of density
        `rho_c` under gravity `grav`.

    Returns
    -------
    eos : function

        The desired equation of state.

    """

    warnings.warn("Replace with `load_eos(eos, '', grav, rho_c)`", DeprecationWarning, 2)
    if callable(eos):
        fn = eos
    else:
        fn = load_eos(eos, "")
    if grav is not None and rho_c is not None:
        fn = make_bsq(fn, grav, rho_c)

    return fn


def make_eos_s_t(eos, grav=None, rho_c=None):
    """Make a function for the partial S and T derivatives of an equation of state

    Parameters
    ----------
    eos, grav, rho_c :
        See `make_eos`

    Returns
    -------
    eos_s_t : function

        Function returning two outputs, namely the partial derivatives with
        respect to its first two arguments (practical / Absolute salinity and
        potential / Conservative temperature) of the desired equation of
        state.
    """

    warnings.warn("Replace with `load_eos(eos, '_s_t', grav, rho_c)`", DeprecationWarning, 2)
    if callable(eos):
        fn = eos
    else:
        fn = load_eos(eos, "_s_t")
    if grav is not None and rho_c is not None:
        fn = make_bsq(fn, grav, rho_c)

    return fn


def make_eos_p(eos, grav=None, rho_c=None):
    """Make a function for the partial P derivative of an equation of state

    Parameters
    ----------
    eos, grav, rho_c :
        See `make_eos`

    Returns
    -------
    eos_p : function

        Function returning the partial derivative with respect to the third
        argument (pressure) of the desired equation of state.
    """

    warnings.warn("Replace with `load_eos(eos, '_p', grav, rho_c)`", DeprecationWarning, 2)
    if callable(eos):
        fn = eos
    else:
        fn = load_eos(eos, "_p")
    if grav is not None and rho_c is not None:
        fn = make_bsq(fn, grav, rho_c)

    return fn


@ft.lru_cache(maxsize=10)
def make_bsq(fn, grav, rho_c):
    """Make a Boussinesq version of a given equation of state (or its partial derivative(s))

    Parameters
    ----------
    fn : function
        Function with (salinity, temperature, pressure, pfac) as inputs.  Typically
        this is the equation of state, returning the density or specific volume.
        However, it can also be a function for partial derivative(s) of the
        equation of state with respect to salinity, temperature, or pressure.
        The 4th argument, `pfac`, pre-multiplies `pressure` before the main calculation,
        and also post-multiplies the output as many times as there are pressure 
        partial derivatives. 

    grav : float
        Gravitational acceleration [m s-2]

    rho_c : float
        Boussinesq reference density [kg m-3]

    Returns
    -------
    fn_bsq : function
        Boussinesq version of `fn`.
        The inputs to `fn_bsq` are (salinity, temperature, depth).
    """

    # Hydrostatic conversion from depth [m] to pressure [dbar]
    z_to_p = 1e-4 * grav * rho_c

    # Get parameters to fn
    params = inspect.signature(fn).parameters

    if len(params) != 4:
        raise ValueError("Expected `fn` to accept 4 arguments.")

    # Get default value of last (4th) parameter, then multiply by z_to_p
    pfac = next(reversed(params.values())).default * z_to_p

    @nb.njit
    def fn_bsq(s, t, z):
        return fn(s, t, z, pfac)

    return fn_bsq


@ft.lru_cache(maxsize=10)
def vectorize_eos(eos):
    """Convert an `eos` function that takes scalar inputs into one taking arrays.

    Parameters
    ----------
    eos : function
        Any function taking three scalar inputs (additional optional arguments
        are allowed but only take their default value) and returning one scalar output,
        such as the equation of state.  Note this does not work for functions
        returning multiple outputs (i.e. a tuple), such as a function returning
        the partial derivatives of the equation of state.

    Returns
    -------
    eos_vec : function
        A `@numba.vectorize`'d version of `eos`, which can take array inputs and
        returns one array output.  The array inputs' shape need not match
        exactly, but must be broadcastable to each other.
    """

    @nb.vectorize
    def eos_vec(s, t, p):
        return eos(s, t, p)

    # suppress RuntimeWarning when NaN's present in `s` array.
    # see https://github.com/numba/numba/issues/4793
    def eos_vec_nowarning(s, t, p):
        with np.errstate(invalid="ignore"):
            return eos_vec(s, t, p)

    return eos_vec_nowarning
