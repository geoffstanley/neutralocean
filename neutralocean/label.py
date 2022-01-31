import numpy as np

from neutralocean.interp1d import make_interpolator, make_kernel


# CHECK VALUE from MATLAB:
# >> veronis_density(0, S(:,i0,j0), T(:,i0,j0), Z, 10, 1500, 1, @ppc_linterp)
# 1027.770044499011
def veronis_density(
    S,
    T,
    P,
    p1,
    p_ref=0.0,
    p0=None,
    dp=1.0,
    interp="linear",
    eos="gsw",
    eos_s_t=None,
    grav=None,
    rho_c=None,
):
    """The surface density plus the integrated vertical gradient of Locally
    Referenced Potential Density

    Determines the Veronis density [1]_ [2]_ at vertical position `p1` on a
    cast with hydrographic properties `(S, T, P)`.  The Veronis density is
    the potential density (referenced to `p_ref`) evaluated at `p0` on the
    cast, plus the integral (dP) of the vertical (d/dP) derivative of Locally
    Referenced Potential Density (LRPD) from `P = p0` to `P = p1`.  The
    vertical (d/dP) derivative of LRPD is `rho_S dS/dP + rho_T dT/dP` where
    `rho_S` and `rho_T` are the partial derivatives of density with respect
    to `S` and `T`, and `dS/dP` and `dT/dP` are the derivatives of `S` and
    `T` with respect to `P` in the water column.  If `p0` or `p1` are outside
    the range of `P`, NaN is returned.

    Parameters
    ----------
    S, T, P : 1D ndarray of float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth of data points on the cast.  `P` must increase
        monotonically along its last dimension.

    p1 : float

        Pressure or depth at which the Veronis density is evaluated

    Returns
    -------
    d : float

        Veronis density

    Other Parameters
    ----------------
    p_ref : float, Default 0.

        reference pressure or depth for potential density

    p0 : float, Default `P[0]`

        Pressure or depth at which the potential density is evaluated

    dp : float, Default 1.

        Maximum interval of pressure or depth in trapezoidal numerical
        integration

    interp : str, Default ``linear''

        Method for vertical interpolation.  Use ``linear'' for linear
        interpolation, and ``pchip'' for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `interp1d`.

    eos : function

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure (if non-Boussinesq) or depth (if Boussinesq).

    eos_s_t : function

        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T`.  The inputs are `S`, `T`, and
        pressure (if non-Boussinesq) or depth (if Boussinesq).

        This need not be @numba.njit decorated but should be vectorized, as it
        will be called a few times with ndarray inputs.

    Notes
    -----
    The result of this function can serve as a density label for an
    approximately neutral surface. However, this is NOT the same as a value
    of the Jackett and McDougall (1997) Neutral Density variable. This is
    true even if you were to provide this function with the same cast that
    Jackett and McDougall (1997) used to initially label their Neutral
    Density variable, namely the cast at 188 deg E, 4 deg S, from the Levitus
    (1982) ocean atlas. Some difference would remain, because of differences
    in numerics, and because of a subsequent smoothing step in the Jackett
    and McDougall (1997) algorithm. This function merely allows one to label
    an approximately neutral surface with a density value that is INTERNALLY
    consistent within the dataset where one's surface lives. This function is
    NOT to compare density values against those from any other dataset, such
    as 1997 Neutral Density.

    .. [1] Veronis, G. (1972). On properties of seawater defined by temperature,
    salinity, and pressure. Journal of Marine Research, 30(2), 227.

    .. [2] Stanley, McDougall, Barker 2021, Algorithmic improvements to finding
     approximately neutral surfaces, Journal of Advances in Earth System
     Modelling, 13(5).
    """

    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)), 'P must be same size as S')
    # assert(isvector(S), 'S, T, P must be 1D. (Veronis density is only useful for one water column at a time!)')
    # assert(isscalar(p0), 'p0 must be a scalar')
    # assert(isscalar(p1), 'p1 must be a scalar')

    # Get interpolation kernels for 0th and 1st derivative
    interp_kers = make_kernel(interp, (0, 1))

    interp_1_twice = make_interpolator(interp, 0, "1", True)

    if p0 is None:
        p0 = P[0]

    if (
        np.isnan(p0)
        or np.isnan(p1)
        or np.isnan(P[0])
        or p0 < P[0]
        or P[-1] < p0
        or p1 < P[0]
        or P[-1] < p1
    ):
        return np.nan

    # P[k0-1] <= p0 <= P[k0]  when  k0 == 1
    # P[k0-1] <  p0 <= P[k0]  when  k0 > 1
    # Similarly for k1.
    k0 = max(1, np.searchsorted(P, p0))
    k1 = max(1, np.searchsorted(P, p1))

    # Integrate from p0 to P[k0]
    d1 = _int_x_k(p0, k0, dp, P, S, T, eos_s_t, interp_kers)

    # Integrate from P[k0] to P[k1]
    for k in range(k0, k1):
        # Integrate from P[k] to P[k+1], for k=k0 to k=k1-1
        d1 += _int_x_k(P[k], k + 1, dp, P, S, T, eos_s_t, interp_kers)

    # Integrate from p1 to P[k1], and subtract this
    d1 -= _int_x_k(p1, k1, dp, P, S, T, eos_s_t, interp_kers)

    # Calculate potential density, referenced to p_ref, at p0
    s0, t0 = interp_1_twice(p0, P, S, T)
    d0 = eos(s0, t0, p_ref)

    return d0 + d1


def _int_x_k(p, k, dp, P, S, T, eos_s_t, interp_kers):
    # Integrate from p to P[k] using trapezoidal integration with spacing dp or smaller

    # If p == P[k], this returns 0.0

    n = np.int(np.ceil((P[k] - p) / dp)) + 1  # points between p and P[k], inclusive
    p_ = np.linspace(p, P[k], n)  # intervals are not larger than dp

    # Use piecewise polynomial coefficients as provided. Be sure to pass the
    # index of this part to avoid any issues evaluating a discontinuous piecewise
    # polynomial (probably the derivative, if either) at the knot.
    s_ = np.zeros(n)
    t_ = np.zeros(n)
    dsdp_ = np.zeros(n)
    dtdp_ = np.zeros(n)
    for i in range(n):
        s_[i] = interp_kers[0](p_[i], P, S, k)
        t_[i] = interp_kers[0](p_[i], P, T, k)
        dsdp_[i] = interp_kers[1](p_[i], P, S, k)
        dtdp_[i] = interp_kers[1](p_[i], P, T, k)

    # To use linear interpolation internally, replace the above lines with the following 7 lines
    # dp = P[k] - P[k-1]
    # dsdp_ = (S[k] - S[k-1]) / dp
    # dtdp_ = (T[k] - T[k-1]) / dp
    # s0 = ( S[k-1] * (P[k] - p) + S[k] * (p - P[k-1]) ) / dp
    # t0 = ( T[k-1] * (P[k] - p) + T[k] * (p - P[k-1]) ) / dp
    # s_ = np.linspace(s0, S[k], n)
    # t_ = np.linspace(t0, T[k], n)

    rs_, rt_ = eos_s_t(s_, t_, p_)
    y_ = rs_ * dsdp_ + rt_ * dtdp_
    return np.trapz(y_, x=p_)


# def veronis_label(p_ref, t_ref, S, T, P, p, pin, eos, eos_s_t, dp=1, interpfn=linear_coeffs):
# 1. Do a temporal neutral_trajectory from (t,i0,j0,p[i0,j0]) to (t_ref,i0,j0,p0)
# 2. Evaluate veronis density at (t_ref, i0, j0, p0)
