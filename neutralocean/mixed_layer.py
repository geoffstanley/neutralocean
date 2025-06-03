""" Mixed Layer """

from .ppinterp import make_pp
from .eos import load_eos, vectorize_eos
from .lib import _process_casts, _process_vert_dim, local_functions

eos_ = load_eos("gsw")  # default


def mld(
    S,
    T,
    P,
    eos=eos_,
    pot_dens_diff=0.03,
    ref_p=100.0,
    bottle_index=1,
    interp="linear",
    vert_dim=-1,
    **kw,
):
    """Calculate the mixed layer pressure or depth

    The mixed layer depth (mld) is the pressure or depth at which the potential density
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

    eos : function, Default `neutralocean.eos.gsw.specvol`

        Function taking three inputs corresponding to (`S, T, P)`, and
        outputting a the in-situ density or specific volume.

    pot_dens_diff : float, Default 0.03

        Difference in potential density [kg m-3] between the near-surface and
        the base of the mixed layer.

    ref_p : float, Default 100.0

        Reference pressure or depth for potential density [dbar or m].

    bottle_index : int, Default 1

        The index for the bottle on each cast where the "near surface" potential
        density is calculated.  Note this index is 0-based.  The Default of 1
        therefore indexes the second bottle in each cast.

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use `'linear'` for linear
        interpolation, and `'pchip'` for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `interp1d`.

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes` section of `potential_surf`.


    Returns
    -------
    mld : ndarray

        A 2D array giving the pressure [dbar] or depth [m, positive] at the base of the
        mixed layer
    """

    rho_c = kw.get("rho_c")
    grav = kw.get("grav")
    if grav is not None or rho_c is not None or isinstance(eos, str):
        raise ValueError(
            "`grav` and `rho_c` and `eos` as a string are no longer supported. "
            "Pass `eos` as a function, which can be obtained from "
            "`neutralocean.load_eos`. See the `examples` folder for examples."
        )

    # Ensure eos is vectorized. It's okay if eos already was.
    eos = vectorize_eos(eos)

    # Make universal interpolator from the interpolation kernel
    interp_fn = make_pp(interp, kind="u", out="interp")

    # Convert vert_dim from str to int if needed
    vert_dim = _process_vert_dim(vert_dim, S)

    # Convert S, T, P from xarray to numpy arrays if needed, and make casts contiguous in memory
    S, T, P = _process_casts(S, T, P, vert_dim)

    SB = S[..., bottle_index : bottle_index + 1]  # keep singleton trailing dim
    TB = T[..., bottle_index : bottle_index + 1]

    # Calculate potential density difference between each data point and the
    # near-surface bottle
    if eos(34.5, 3, 1000) > 1:  # eos computes in-situ density
        DD = eos(S, T, ref_p) - eos(SB, TB, ref_p)
    else:  # eos computes specific volume
        DD = 1 / eos(S, T, ref_p) - 1 / eos(SB, TB, ref_p)

    # Find the pressure or depth at which the potential density difference
    # exceeds the threshold pot_dens_diff
    return interp_fn(pot_dens_diff, DD, P, 0)


__all__ = local_functions(locals(), __name__)
