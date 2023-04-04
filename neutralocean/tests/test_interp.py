import numpy as np
import pytest

from neutralocean.interp1d import make_interpolator
from neutralocean.ppinterp import make_pp, ppval, valid_range_1
from scipy.interpolate import UnivariateSpline, PchipInterpolator

N = 4  # number of 1D interpolation problems
K = 10  # number of grid points in each interpolation problem

# Monotonic but non-uniform independent data
X1 = np.linspace(0, 10, K) ** 1.2
X = np.tile(X1, (N, 1))

# Build dependent data
Y = np.empty((N, K), dtype=np.float64)
Y[0:2] = np.ones(K) * (X1 / X1[-1]) ** 2  # steadily increasing
Y[2] = np.sin(X1 / X1[-1] * 2 * np.pi)  # smooth wave
Y[3] = Y[1] * np.cos(X1 / X1[-1] * 10 * np.pi)  # crazy wave, with some NaN's
Y[3, -3:] = np.nan

X[1, 1:] = np.nan  # test an extreme case of nan data, e.g. 1 ocean cell


# Interpolate between each knot
x_midpts = X1[0:-1] + np.diff(X1) / 2

# Interpolate to each knot point and each midpoint
x_targets = np.sort(np.concatenate((X1, x_midpts)))

# Note re: lhs vs rhs:
# Our code won't agree with SciPy when evaluating a function that is piecewise
# discontinuous (e.g. first derivative of a linear interpolant or second
# derivative of a PHCIP) at the knots, because we evaluate using the "left"
# side whereas SciPy evaluates using the "right" side.
@pytest.mark.parametrize(
    "interp,num_deriv,x",
    [
        ("linear", 0, x_targets),
        ("linear", 1, x_midpts),  # see "Note re: lhs vs rhs"
        ("pchip", 0, x_targets),
        ("pchip", 1, x_targets),
        ("pchip", 2, x_midpts),  # see "Note re: lhs vs rhs"
    ],
)
def test_interp(interp, num_deriv, x):

    # Interpolate with SciPy
    y = np.full((N, x.size), np.nan, dtype=float)
    for i in range(N):
        k, K = valid_range_1(X[i] + Y[i])
        # k = min(find_first_nan(Y[i]), find_first_nan(X[i]))
        try:
            if interp == "linear":
                fn = UnivariateSpline(
                    X[i, k:K], Y[i, k:K], k=1, s=0, ext="raise"
                )
            elif interp == "pchip":
                fn = PchipInterpolator(X[i, k:K], Y[i, k:K], extrapolate=False)
            fn = fn.derivative(num_deriv)
            for j in range(x.size):
                y[i, j] = fn(x[j])
        except:
            # extrapolation was needed (only for UnivariateSpline)
            # or not enough valid data points (e.g. X has 1 non-nan value)
            pass  # leave as nan

    # Interpolate with our methods: first, on the fly using interp1d
    interp_fn = make_interpolator(interp, num_deriv, "u")
    y1 = np.empty_like(y)
    for j in range(x.size):
        y1[:, j] = interp_fn(x[j], X, Y)

    assert np.allclose(y, y1, equal_nan=True)

    # Interpolate with our methods:
    # second, with piecewise polynomial coefficients, using ppinterp
    ppc_fn = make_pp(interp, kind="u", out="coeffs")
    Yppc = ppc_fn(X, Y)
    y2 = np.empty_like(y)
    for j in range(x.size):
        y2[:, j] = ppval(x[j], X, Yppc, num_deriv)

    # PCHIPs have machine precision differences between interp1d and ppinterp.
    # assert np.array_equal(y1, y2, equal_nan=True)

    assert np.allclose(y1, y2, equal_nan=True)
