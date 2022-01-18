""" Mixed Layer """

from neutralocean.interp_ppc import linear_coeffs, interp
from neutralocean.eos.tools import vectorize_eos
from neutralocean.lib import _process_casts, _process_vert_dim


def mixed_layer(
    S,
    T,
    P,
    eos,
    pot_dens_diff=0.03,
    ref_p=100.0,
    bottle_index=1,
    interp_fn=linear_coeffs,
    vert_dim=-1,
):
    """Calculate the pressure or depth at the bottom of the mixed layer

    The mixed layer is the pressure or depth at which the potential density
    (referenced to `ref_p`) exceeds the potential density near the surface
    (specifically, the bottle indexed by `bottle_index` on each cast) by an
    amount of `pot_dens_diff`.

    Parameters
    ----------
    S, T : ndarray or xarray.DataArray

        3D practical / Absolute salinity and potential / Conservative
        temperature --- that is, a 2D array of 1D water columns or "casts"

    P : ndarray or xarray.DataArray

        In the non-Boussinesq case, `P` is the 3D pressure, sharing the same
        dimensions as `S` and `T`.

        In the Boussinesq case, `P` is the depth and can be 3D with the same
        structure as `S` and `T`, or can be 1D with as many elements as there
        are in the vertical dimension of `S` and `T`.

        Must increase monotonically along the first dimension (i.e. downwards).

    eos : function

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and `P` inputs.

        This function should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    pot_dens_diff : float, Default 0.03

        Difference in potential density [kg m-3] between the near-surface and
        the base of the mixed layer.

    ref_p : float, Default 100.0

        Reference pressure or depth for potential density [dbar or m].

    bottle_index : int, Default 1

        The index for the bottle on each cast where the "near surface" potential
        density is calculated.  Note this index is 0-based.  The Default of 1
        therefore indexes the second bottle in each cast.

    interp_fn : function, Default `linear_coeffs`

        Function that calculates coefficients of piecewise polynomial
        interpolants of `S` and `T` as functions of `P`.  Options include
        ``linear_coeffs`` and ``pchip_coeffs`` from ``interp_ppc.py``.

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes`.


    Returns
    -------
    ML : ndarray

        A 2D array giving the pressure [dbar] or depth [m, positive] of the mixed layer
    """

    # Ensure eos is vectorized. It's okay if eos already was.
    eos = vectorize_eos(eos)

    # Convert vert_dim from str to int if needed
    vert_dim = _process_vert_dim(vert_dim, S)

    # Convert S, T, P from xarray to numpy arrays if needed, and make casts contiguous in memory
    S, T, P = _process_casts(S, T, P, vert_dim)

    SB = S[:, :, bottle_index : bottle_index + 1]  # retain singleton trailing dimension
    TB = T[:, :, bottle_index : bottle_index + 1]

    # Calculate potential density difference between each data point and the
    # near-surface bottle
    if eos(34.5, 3, 1000) > 1:  # eos computes in-situ density
        DD = eos(S, T, ref_p) - eos(SB, TB, ref_p)
    else:  # eos computes specific volume
        DD = 1 / eos(S, T, ref_p) - 1 / eos(SB, TB, ref_p)

    # Find the pressure or depth at which the potential density difference
    # exceeds the threshold pot_dens_diff
    Pppc = interp_fn(DD, P)
    return interp(pot_dens_diff, DD, P, Pppc)
