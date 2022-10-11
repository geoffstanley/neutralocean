import functools as ft
import numba as nb
from numba import f4, f8

from .interp1d import _interp_1, _interp_1_YZ, _interp_1_fg, _interp_1_fg_YZ
from .interp1d import _interp_n, _interp_n_YZ, _interp_n_fg, _interp_n_fg_YZ
from .linear import _linterp, _linterp1
from .pchip import _pchip, _pchip1, _pchip2, _pchip3


@ft.lru_cache
def make_interpolator(interpolant="linear", deriv=0, kind="u", two=False):
    """Factory function to build various interpolating functions.

    Parameters
    ----------
    interpolant : str, Default "linear"

        - If "linear", build a function for linear interpolation.
        - If "pchip", build a function for Piecewise Cubic Hermite Interpolating
          Polynomial interpolation.

    deriv : int or tuple of int, Default 0

        Build a function that returns the `deriv` derivative of the
        `interpolant`.
        Note that `deriv = 0` simply builds a function that interpolates
        according to `interpolant`.
        If a tuple, gives a function that returns multiple values, one for each
        of the derivatives in this tuple.
        Currently, tuples of length 3 or more are not supported.

    kind : str, Default "u"

        - If "1", a 'numba.njit'ed function is returned that does 1 interpolation.
          The function's inputs are (see `_interp_1`)

            - x : float
            - X : ndarray(float, 1d)
            - Y : ndarray(float, 1d)
            - Z : ndarray(float, 1d) -- only if `two` is True.

        - If "n", a 'numba.njit'ed function is returned that does many
          interpolations of dependent data to a single evaluation site on a single
          independent data array.  The function's inputs are (see `_interp_n`)

            - x : float
            - X : ndarray(float, 1d)
            - Y : ndarray(float, nd)
            - Z : ndarray(float, nd) -- only if `two` is True; must have Z.shape == Y.shape

        - If "u", a universal function is returned, whose inputs can be
          multidimensional numpy arrays, so long as they are appropriately
          broadcastable.  The function's inputs are

            - x : ndarray(float, (n-1)d)
            - X : ndarray(float, nd)
            - Y : ndarray(float, nd)
            - Z : ndarray(float, nd) -- only if `two` is True.

          If all dimensions are full, then the dimensions of `x`  match those of
          `X` less its last dimension, and all of `X`, `Y`, `Z` have the same
          dimension.
          However, these inputs dimensions can be a subset of their "full"
          dimensions above.  E.g. `X` can be a 1D array of length equal to the
          size of the last dimension of `Y`.
          This is achieved via `numba.guvectorize`ing the relevant `_interp_1_*`
          function.
          Most users will want this "u" option, as it can do everything the "1" or
          "n" options can do.  The exception is when a `@numba.njit` function is
          explicitly needed, such as when this is called inside another
          `@numba.njit` function.

    two : bool, Default False

        If True, returns an interpolating function that takes four inputs, the
        last two of which are two dependent data arrays, `Y` and `Z`.
        If False, just one dependent data array, `Y`, is input to the returned
        function.

    Returns
    -------
    f : function

        An interpolating function, taking three inputs,
            - `x`, the evaluation site,
            - `X`, the independent data, and
            - `Y`, the dependent data

        or a fourth input when `two` is True,
            - `Z`, another set of dependent data

        This function's output is determined by the inputs `deriv` and `two`.
        
        For example, with `deriv` = 0 and `two` = False, the output is
            - `y`, the interpolant of `Y` as a function of `X` evaluated at `x`.

        If `deriv` = 1 and `two` = True, then the outputs are
            - `y`, the 1st derivative for `Y` as a function of `X` evaluated at `x`,
            - `z`, the 1st derivative for `Z` as a function of `X` evaluated at `x`.

        If `deriv` = (0,1) and `two` = True, then the function's output is
            - `y0`, the interpolant of `Y` as a function of `X` evaluated at `x`,
            - `z0`, the interpolant of `Z` as a function of `X` evaluated at `x`,
            - `y1`, the 1st derivative for `Y` as a function of `X` evaluated at `x`,
            - `z1`, the 1st derivative for `Z` as a function of `X` evaluated at `x`.

    Examples
    --------
    >>> # Make fake pressure, salinity and temperature data
    >>> P = np.linspace(0, 4500, 10)
    >>> S = np.reshape([0, 1], (-1, 1)) + np.linspace(33.0, 35.0, 10).reshape((1, -1))
    >>> T = np.reshape([0, 1], (-1, 1)) + np.linspace(29.0, -2.0, 10).reshape((1, -1))
    >>> # Build a universal linear interpolator for two variables
    >>> interp = make_interpolator("linear", 0, "u", True)
    >>> p = 400.0  # interpolate to an isobaric surface (constant P)
    >>> s, t = interp(p, P, S, T)  # interpolate S and T to P = p.
    >>> p = [400.0, 450.0] # interpolate to a surface with varying pressure
    >>> s, t = interp(p, P, S, T)
    """

    ker0 = make_kernel(interpolant, deriv)

    if isinstance(ker0, tuple):
        lenker = len(ker0)
        if lenker == 2:
            ker1 = ker0[1]
        ker0 = ker0[0]  # Need to dereference these for numba
    else:
        lenker = 1

    if kind == "1":
        if lenker == 1:
            if not two:

                @nb.njit
                def fcn(x, X, Y):
                    return _interp_1(ker0, x, X, Y)

            else:

                @nb.njit
                def fcn(x, X, Y, Z):
                    return _interp_1_YZ(ker0, x, X, Y, Z)

        elif lenker == 2:
            if not two:

                @nb.njit
                def fcn(x, X, Y):
                    return _interp_1_fg(ker0, ker1, x, X, Y)

            else:

                @nb.njit
                def fcn(x, X, Y, Z):
                    return _interp_1_fg_YZ(ker0, ker1, x, X, Y, Z)

    elif kind == "n":

        if lenker == 1:
            if not two:

                @nb.njit
                def fcn(x, X, Y):
                    return _interp_n(ker0, x, X, Y)

            else:

                @nb.njit
                def fcn(x, X, Y, Z):
                    return _interp_n_YZ(ker0, x, X, Y, Z)

        elif lenker == 2:
            if not two:

                @nb.njit
                def fcn(x, X, Y):
                    return _interp_n_fg(ker0, ker1, x, X, Y)

            else:

                @nb.njit
                def fcn(x, X, Y, Z):
                    return _interp_n_fg_YZ(ker0, ker1, x, X, Y, Z)

    elif kind == "u":

        if lenker == 1:
            if not two:

                @nb.guvectorize(
                    [
                        (f4, f4[:], f4[:], f4[:]),
                        (f8, f8[:], f8[:], f8[:]),
                    ],
                    "(),(n),(n)->()",
                    nopython=True,
                )
                def fcn(x, X, Y, y):
                    y[0] = _interp_1(ker0, x, X, Y)

            else:

                @nb.guvectorize(
                    [
                        (f4, f4[:], f4[:], f4[:], f4[:], f4[:]),
                        (f8, f8[:], f8[:], f8[:], f8[:], f8[:]),
                    ],
                    "(),(n),(n),(n)->(),()",
                    nopython=True,
                )
                def fcn(x, X, Y, Z, y, z):
                    y[0], z[0] = _interp_1_YZ(ker0, x, X, Y, Z)

        elif lenker == 2:
            if not two:

                @nb.guvectorize(
                    [
                        (f4, f4[:], f4[:], f4[:], f4[:]),
                        (f8, f8[:], f8[:], f8[:], f8[:]),
                    ],
                    "(),(n),(n)->(),()",
                    nopython=True,
                )
                def fcn(x, X, Y, yf, yg):
                    yf[0], yg[0] = _interp_1_fg(ker0, ker1, x, X, Y)

            else:

                @nb.guvectorize(
                    [
                        (f4, f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:]),
                        (f8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:]),
                    ],
                    "(),(n),(n),(n)->(),(),(),()",
                    nopython=True,
                )
                def fcn(x, X, Y, Z, yf, zf, yg, zg):
                    yf[0], zf[0], yg[0], zg[0] = _interp_1_fg_YZ(ker0, ker1, x, X, Y, Z)

    else:
        raise ValueError(f"Expected kind in ('u', '1', 'n'); got {kind}")

    return fcn


# Does not work.  Cannot index a tuple inside the guvectorize, it seems...
# def make_d(*f):
#     n = len(f)

#     @nb.guvectorize(
#         [(f8, f8[:], f8[:], nb.int64, f8[:])],
#         "(),(n),(n),()->()",
#     )
#     def fcn(x, X, Y, d, y):
#         if d < n:
#             y[0] = interp_1(x, X, Y, f[d])
#         else:
#             y[0] = 0.0

#     return fcn


def make_kernel(interpolant, deriv):
    """
    Select the interpolating kernel(s) for a given interpolation method and derivative.

    Parameters
    ----------
    interpolant : str, Default "linear"

        Returns interpolating kernel(s) for
            linear interpolation if "linear"
            Piecewise Cubic Hermite Interpolating Polynomial interpolation if "pchip"

    deriv : int or tuple of int, Default 0

        Return interpolating kernel for the `deriv` derivative of the `interpolant`.
        Note that `deriv = 0` simply builds a function that interpolates
        according to `interpolant`.
        If a tuple, returns a tuple of interpolating kernels, one for each
        of the derivatives in this tuple.

    """

    if interpolant == "linear":
        kers = (_linterp, _linterp1)
    elif interpolant == "pchip":
        kers = (_pchip, _pchip1, _pchip2, _pchip3)
    else:
        raise ValueError(
            f"Expected `interpolant` in ('linear', 'pchip'); got {interpolant}"
        )

    if isinstance(deriv, (tuple, list)):
        ker = tuple(kers[d] for d in deriv)
    else:
        ker = kers[deriv]

    return ker
