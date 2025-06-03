import numpy as np
import numba as nb

from .ppinterp import make_pp, ppval_1_two, pval
from .eos import load_eos
from .lib import local_functions

def veronis(
    p1,
    S,
    T,
    P,
    p0=None,
    p_ref=0.0,
    b=1.0,
    dp=1.0,
    interp="linear",
    eos=None,
    eos_s_t=None,
    **kw,
):
    """The surface density plus the integrated vertical gradient of Locally
    Referenced Potential Density

    Determines the Veronis density [1]_ [2]_ at vertical position `p1` on a cast
    with hydrographic properties `(S, T, P)`. The Veronis density is the
    potential density (referenced to `p_ref`) evaluated at `p0` on the cast,
    plus `b` times the integral (obtained numerically by trapezoidal
    integration with step sizes of `dP` or less) of the vertical
    (d/dP) derivative of Locally Referenced Potential Density (LRPD) from `P =
    p0` to `P = p1`. The vertical (d/dP) derivative of LRPD is `rho_S dS/dP +
    rho_T dT/dP` where `rho_S` and `rho_T` are the partial derivatives of
    density with respect to `S` and `T`, and `dS/dP` and `dT/dP` are the
    derivatives of `S` and `T` with respect to `P` in the water column.  If
    `p0` or `p1` are outside the range of `P`, NaN is returned.

    In common oceanographic mathematical notation, the above is written
    as (see Equation (A2) in [2]_)
        ρᵥ(p₁) = ρ(S(p₀), Θ(p₀), p_ref) + b ∫ [f dS/dp + g dΘ/dp] dp
    where the integral goes from p₀ to p₁, and where
    f(p) = (∂ρ/∂S)(S(p),Θ(p),p) and g(p) = (∂ρ/∂Θ)(S(p),Θ(p),p) are
    the partial derivatives of the equation of state ρ.
    Note the Conservative (or potential) Temperature Θ is denoted `T` in the code.


    Parameters
    ----------
    p1 : float

        Pressure or depth at which the Veronis density is evaluated

    S, T, P : 1D ndarray of float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth of data points on the cast.  `P` must increase
        monotonically along its last dimension.

    Returns
    -------
    d : float

        Veronis density

    Other Parameters
    ----------------
    p0 : float, Default `P[0]`

        Pressure or depth at which the potential density is evaluated

    p_ref : float, Default 0.0

        reference pressure or depth for potential density

    b : float, Default 1.0

        Factor by which to multiply the integral.

    dp : float, Default 1.0

        Maximum interval of pressure or depth in trapezoidal numerical
        integration

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use `'linear'` for linear
        interpolation, and `'pchip'` for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `ppinterp`.

    eos : function, Default `neutralocean.eos.gsw.specvol`

        Function taking three inputs corresponding to (`S, T, P)`, and
        outputting the in-situ density or specific volume.

    eos_s_t : function, `neutralocean.eos.gsw.specvol_s_t`

        Function taking three inputs corresponding to (`S, T, P)`, and
        outputting a tuple containing the partial derivatives of the equation of
        state with respect to `S` and `T`.

        The function(s) should be `@numba.njit` decorated and need not be vectorized
        -- they will be called many times with scalar inputs.

    Notes
    -----
    The Veronis density at a particular pressure / depth on a reference cast can
    serve as a neutral density label for an approximately neutral surface
    (such as an omega surface) intersecting that point, and for temporal
    neutral trajectories through that point, and for other approximately
    neutral surfaces (such as omega surfaces) intersecting that temporal
    neutral trajectory. The Veronis density serves as a useful neutral density
    label in this way because it perfectly represents the stratification in a
    single vertical cast, but its horizontal variations are not meaningful.

    However, the neutral density label obtained by this Veronis density is NOT
    the same as a value of the Jackett and McDougall (1997) [3]_ Neutral Density
    variable. These two values are different even if you were to provide this
    function with the same cast that Jackett and McDougall (1997) used to
    initially label their Neutral Density variable, namely the cast at 188
    deg E, 4 deg S, from the Levitus (1982) ocean atlas. Some difference
    would remain, because of differences in numerics, and because of a
    subsequent smoothing step in the Jackett and McDougall (1997) algorithm.
    This function merely allows one to label an approximately neutral surface
    with a density value that is INTERNALLY consistent within the dataset
    where one's surface lives. This function is NOT to compare density values
    against those from any other dataset, such as 1997 Neutral Density.

    See Appendix A of [2]_ for further information.


    Examples
    --------
    >>> S = np.linspace(34, 35.5, 20)
    >>> T = np.linspace(18, 0, 20)
    >>> P = np.linspace(0, 4000, 20)
    >>> veronis(2000, S, T, P)
    0.0009737571001312298

    Calculate the Veronis density at 2000 dbar on a water column of linearly
    varying salinity and potential temperature.

    .. [1] Veronis, G. (1972). On properties of seawater defined by temperature,
       salinity, and pressure. Journal of Marine Research, 30(2), 227.

    .. [2] Stanley, G. J., McDougall, T. J., & Barker, P. M. (2021). Algorithmic
       Improvements to Finding Approximately Neutral Surfaces. Journal of
       Advances in Modeling Earth Systems, 13(5), e2020MS002436.

    .. [3] Jackett, D. R., & McDougall, T. J. (1997). A neutral density variable
       for the world’s oceans. Journal of Physical Oceanography, 27 (2), 237–263.

    """

    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)), 'P must be same size as S')
    # assert(isvector(S), 'S, T, P must be 1D. (Veronis density is only useful for one water column at a time!)')
    # assert(isscalar(p0), 'p0 must be a scalar')
    # assert(isscalar(p1), 'p1 must be a scalar')
    
    rho_c = kw.get("rho_c")
    grav = kw.get("grav")
    if grav is not None or rho_c is not None or isinstance(eos, str):
        raise ValueError(
            "`grav` and `rho_c` and `eos` as a string are no longer supported. "
            "Pass `eos` and `eos_s_t` as functions, which can be obtained from "
            "`neutralocean.load_eos`. See the `examples` folder for examples."
        )

    if eos is None and eos_s_t is None:
        eos = load_eos("gsw")
        eos_s_t = load_eos("gsw", "_s_t")

    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=True)
    Sppc = ppc_fn(P, S)
    Tppc = ppc_fn(P, T)
    if p0 is None:
        p0 = P[0]

    return _veronis(p1, p0, p_ref, b, P, Sppc, Tppc, dp, eos, eos_s_t)


@nb.njit
def _veronis(p1, p0, p_ref, b, P, Sppc, Tppc, dp, eos, eos_s_t):
    # fast Veronis density, with pre-computed PPC coefficients and with
    # eos and eos_s_t as (njit'ed) functions.
    d0 = pot_dens_1(p0, p_ref, P, Sppc, Tppc, eos)
    return _veronis_affine(p1, p0, d0, b, P, Sppc, Tppc, dp, eos_s_t)


@nb.njit
def _veronis_affine(p1, p0, d0, b, P, Sppc, Tppc, dp, eos_s_t):
    # Affine transformation of the integral of the vertical gradient of LRPD
    return d0 + b * _int_dlrpd_dz(p1, p0, P, Sppc, Tppc, dp, eos_s_t)


@nb.njit
def pot_dens_1(p0, p_ref, P, Sppc, Tppc, eos):
    # Potential density referenced to p_ref, at p0, in 1 cast.
    # Note: NaNs are allowed in `P`, `Sppc`, `Tppc`.
    s0, t0 = ppval_1_two(p0, P, Sppc, Tppc)
    return eos(s0, t0, p_ref)


@nb.njit
def _int_dlrpd_dz(p1, p0, P, Sppc, Tppc, dp, eos_s_t):
    # Integrate d(LRPD)/dz from p0 to p1 in 1 cast.
    if np.isnan(p0 + p1 + P[0]) or p0 < P[0] or P[-1] < p0 or p1 < P[0] or P[-1] < p1:
        return np.nan

    if p1 >= p0:
        sgn = 1
    else:
        sgn = -1
        p1, p0 = p0, p1

    # P[k0-1] <= p0 <= P[k0]  when  k0 == 1
    # P[k0-1] <  p0 <= P[k0]  when  k0 > 1
    # Similarly for k1.
    k0 = max(1, np.searchsorted(P, p0))
    k1 = max(1, np.searchsorted(P, p1))

    # Integrate from p0 to P[k0]
    d1 = _int_x_k(p0, k0 - 1, dp, P, Sppc, Tppc, eos_s_t)

    # Integrate from P[k0] to P[k1]
    for k in range(k0, k1):
        # Integrate from P[k] to P[k+1], for k=k0 to k=k1-1
        d1 += _int_x_k(P[k], k, dp, P, Sppc, Tppc, eos_s_t)

    # Integrate from p1 to P[k1], and subtract this
    d1 -= _int_x_k(p1, k1 - 1, dp, P, Sppc, Tppc, eos_s_t)

    return d1 * sgn


@nb.njit
def _int_x_k(p, k, dp, P, Sppc, Tppc, eos_s_t):
    # Integrate from p to P[k+1] using trapezoidal integration with spacing dp or smaller

    # If p == P[k+1], this returns 0.0

    # Number of points between p and P[k], inclusive
    n = int(np.ceil((P[k + 1] - p) / dp)) + 1

    p_ = np.linspace(p, P[k + 1], n)  # intervals are not larger than dp

    # Use piecewise polynomial coefficients as provided. Be sure to pass the
    # index of this part to avoid any issues evaluating a discontinuous piecewise
    # polynomial (probably the derivative, if either) at the knot.
    s_ = np.zeros(n)
    t_ = np.zeros(n)
    dsdp_ = np.zeros(n)
    dtdp_ = np.zeros(n)
    for i in range(n):
        dx = p_[i] - P[k]
        s_[i] = pval(dx, Sppc[k], 0)
        t_[i] = pval(dx, Tppc[k], 0)
        dsdp_[i] = pval(dx, Sppc[k], 1)
        dtdp_[i] = pval(dx, Tppc[k], 1)

    # To use linear interpolation internally, replace the above lines with the following 7 lines
    # dP = P[k+1] - P[k]
    # dsdp_ = (S[k+1] - S[k]) / dP
    # dtdp_ = (T[k+1] - T[k]) / dP
    # s0 = ( S[k] * (P[k+1] - p) + S[k+1] * (p - P[k]) ) / dP
    # t0 = ( T[k] * (P[k+1] - p) + T[k+1] * (p - P[k]) ) / dP
    # s_ = np.linspace(s0, S[k+1], n)
    # t_ = np.linspace(t0, T[k+1], n)

    rs_, rt_ = eos_s_t(s_, t_, p_)
    y_ = rs_ * dsdp_ + rt_ * dtdp_
    return np.trapz(y_, x=p_)


# TODO
# def veronis_label(p_ref, t_ref, S, T, P, p, pin, eos, eos_s_t, dp=1, interpfn=linear_coeffs):
# 1. Do a temporal neutral_trajectory from (t,i0,j0,p[i0,j0]) to (t_ref,i0,j0,p0)
# 2. Evaluate veronis density at (t_ref, i0, j0, p0)

__all__ = local_functions(locals(), __name__)
