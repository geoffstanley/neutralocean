"""Tools for handling the Equation of State"""
import functools as ft
import numba as nb
import importlib

# List of modules in the same directory as this file, each of which must have
# the following numba.njit'ed functions:  rho, rho_s_t, rho_p.
modules = {"gsw": "specvol", "jmd95": "rho", "jmdfwg06": "rho"}


@ft.lru_cache(maxsize=10)
def _make_eos(eos, derivs, num_p_derivs=0, grav=None, rho_c=None):
    if isinstance(eos, str):
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

    elif callable(eos):
        fn = eos

    else:
        raise TypeError(
            f"Since eos was not a string, eos must be a callalbe function; found {eos}"
        )

    if grav != None and rho_c != None:
        fn = make_bsq(fn, grav, rho_c, num_p_derivs)

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

    .. [1] McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and
       the Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127,
       SBN 978-0-646-55621-5.

    .. [2] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388

    .. [3] Jackett, D. R., McDougall, T. J., Feistel, R., Wright, D. G., &
       Griffies, S. M. (2006). Algorithms for Density, Potential Temperature,
       Conservative Temperature, and the Freezing Temperature of Seawater.
       Journal of Atmospheric and Oceanic Technology, 23(12), 1709–1728.
       https://doi.org/10.1175/JTECH1946.1
    """

    return _make_eos(eos, "", 0, grav, rho_c)


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

    return _make_eos(eos, "_s_t", 0, grav, rho_c)


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

    return _make_eos(eos, "_p", 1, grav, rho_c)


@ft.lru_cache(maxsize=10)
def make_bsq(fn, grav, rho_c, num_p_derivs=0):
    """Make a Boussinesq version of a given equation of state (or its partial derivative(s))

    Parameters
    ----------
    fn : function
        Function with (salinity, temperature, pressure) as inputs.  Typically
        this is the equation of state, returning the density or specific volume.
        However, it can also be a function for partial derivative(s) of the
        equation of state with respect to salinity, temperature, or pressure.

    grav : float
        Gravitational acceleration [m s-2]

    rho_c : float
        Boussinesq reference density [kg m-3]

    num_p_derivs : int, Default 0
        Number of `p` partial derivatives that relate `fn` to the equation of
        state.  For example,

        - if `fn` is the equation of state, or its partial derivative (of
          any order, with respect to salinity or temperature, pass 0.
        - if `fn` is the partial derivative of the equation of state with
          respect to pressure, pass 1.
        - if `fn` is the second partial derivative of the equation of state
          with respect to salinity and pressure (i.e. ∂²ρ/∂S∂p), pass 1.

    Returns
    -------
    fn_bsq : function
        Boussinesq version of `fn`.
        The inputs to `fn_bsq` are (salinity, temperature, depth).
    """

    # Hydrostatic conversion from depth [m] to pressure [dbar]
    z_to_p = 1e-4 * grav * rho_c

    if num_p_derivs == 0:
        # Slight optimization for later: don't multiply by factor when factor == 1
        @nb.njit
        def fn_bsq(s, t, z):
            return fn(s, t, z * z_to_p)

    else:
        factor = z_to_p**num_p_derivs

        @nb.njit
        def fn_bsq(s, t, z):
            return fn(s, t, z * z_to_p) * factor

    return fn_bsq


@ft.lru_cache(maxsize=10)
def vectorize_eos(eos):
    """Convert an `eos` function that takes scalar inputs into one taking arrays.

    Parameters
    ----------
    eos : function
        Any function taking three scalar inputs and returning one scalar output,
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

    return eos_vec
