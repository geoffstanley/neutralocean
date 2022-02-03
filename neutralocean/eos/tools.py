"""Tools for handling the Equation of State"""
import functools as ft
import numba as nb

from .jmd95 import rho as rho_jmd95
from .jmd95 import rho_s_t as rho_s_t_jmd95
from .jmd95 import rho_p as rho_p_jmd95

from .jmdfwg06 import rho as rho_jmdfwg06
from .jmdfwg06 import rho_s_t as rho_s_t_jmdfwg06
from .jmdfwg06 import rho_p as rho_p_jmdfwg06

from .gsw import rho as rho_gsw
from .gsw import rho_s_t as rho_s_t_gsw
from .gsw import rho_p as rho_p_gsw


def _make_eos(eos, eos_dict, grav=None, rho_c=None, num_p_derivs=0):
    if isinstance(eos, str) and eos in eos_dict.keys():
        eos = eos_dict[eos]

        if grav != None and rho_c != None:
            eos = make_bsq(eos, grav, rho_c, num_p_derivs)

    if not callable(eos):
        raise TypeError(
            f'First input must be a function, "gsw", "jmd95", or "jmdfwg06"; found {eos}'
        )

    return eos


def make_eos(eos, grav=None, rho_c=None):
    """Make an equation of state function, possibly modified for Boussinesq

    Parameters
    ----------
    eos : str or function

        If a str, can be 'gsw' to generate the TEOS-10 specific volume,
        'jmd95' to generate the Jackett and McDougall (1995) in-situ
        density [1]_, or 'jmdfwg06' to generate the Jackett et al (2006)
        in-situ density [2]_.

        If a function, should be an equation of state as a function of
        practical / Absolute salinity, potential / Conservative temperature,
        and pressure (for non-Boussesinq) / depth (for Boussinesq).

    grav, rho_c : float

        Gravitational acceleration [m s-2] and Boussinesq reference density
        [kg m-3]. If both are provided, the equation of state is modified as
        appropriate for the Boussinesq approximation, in which the third
        argument is depth, not pressure. Specifically, a depth `z` is
        converted to ``1e-4 * grav * rho_c * z``, which is the hydrostatic
        pressure [dbar] at depth `z` [m] caused by a water column of density
        `rho_c` under gravity `grav`.

    Returns
    -------
    eos : function

        The desired equation of state.

    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388

    .. [2] Jackett, D. R., McDougall, T. J., Feistel, R., Wright, D. G., &
       Griffies, S. M. (2006). Algorithms for Density, Potential Temperature,
       Conservative Temperature, and the Freezing Temperature of Seawater.
       Journal of Atmospheric and Oceanic Technology, 23(12), 1709–1728.
       https://doi.org/10.1175/JTECH1946.1
    """
    eos_dict = {"jmd95": rho_jmd95, "jmdfwg06": rho_jmdfwg06, "gsw": rho_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c)


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
    eos_dict = {
        "jmd95": rho_s_t_jmd95,
        "jmdfwg06": rho_s_t_jmdfwg06,
        "gsw": rho_s_t_gsw,
    }
    return _make_eos(eos, eos_dict, grav, rho_c)


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
    eos_dict = {"jmd95": rho_p_jmd95, "jmdfwg06": rho_p_jmdfwg06, "gsw": rho_p_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c, 1)


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
        any order) with respect to salinity or temperature, pass 0.
        - if `fn` is the partial derivative of the equation of state with
          respect to pressure, pass 1.
        - if `fn is the second partial derivative of the equation of state
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
        factor = z_to_p ** num_p_derivs

        @nb.njit
        def fn_bsq(s, t, z):
            return fn(s, t, z * z_to_p) * factor

    return fn_bsq


@ft.lru_cache(maxsize=10)
def vectorize_eos(eos):
    """Convert an eos function that takes scalar inputs into one taking arrays

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
        A @numba.vectorize'd version of `eos`, which can take array inputs and
        returns one array output.  The array inputs' shape need not match
        exactly, but must be broadcastable to each other.
    """

    @nb.vectorize
    def eos_vec(s, t, p):
        return eos(s, t, p)

    return eos_vec
