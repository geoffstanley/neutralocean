import numpy as np
import pytest

from neutralocean.ppinterp import make_pp, ppval, valid_range_1
from scipy.interpolate import UnivariateSpline, PchipInterpolator

NX = 6  # number of 1D interpolation problems
NY = 4
K = 10  # number of grid points in each interpolation problem

X1 = np.linspace(0, 10, K) ** 1.2  # Monotonic but non-uniform independent data
X = np.tile(X1, (NX, NY, 1))
X[1, :, 7:K] = np.nan  # Test NaN's at end
X[2, :, 1:] = np.nan  # Test almost all NaN's, except X[0]
X[3, :, 0:2] = np.nan  # Test NaN's at start, ...
X[3, :, 7:K] = np.nan  # ... and end
X[4, :, 0:5] = np.nan  # Test 2 valid points, ...
X[4, :, 7:K] = np.nan  # ... in the middle
X[5, :, :] = np.nan  # Test all NaN

# Build dependent data
Y = np.empty((NX, NY, K), dtype=np.float64)
Y[:, 0:2] = np.ones(K) * (X1 / X1[-1]) ** 2  # steadily increasing
Y[:, 2] = np.sin(X1 / X1[-1] * 2 * np.pi)  # smooth wave

Y[:, 3] = Y[:, 1] * np.cos(X1 / X1[-1] * 10 * np.pi)  # crazy wave ...
Y[:, 3, -3:] = np.nan  # ... with NaN's at end

# Reshape all casts into a 1D array of 1D casts
N = NX * NY
X, Y = [np.reshape(Z, (N, K)) for Z in (X, Y)]

# Interpolate between each knot
x_midpts = X1[0:-1] + np.diff(X1) / 2

# Interpolate to each knot point and each midpoint
x_targets = np.sort(np.concatenate((X1, x_midpts)))

Y1 = Y[0, :]

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
        try:
            if interp == "linear":
                fn = UnivariateSpline(
                    X[i, k:K], Y[i, k:K], k=1, s=0, ext="raise"
                )
            elif interp == "pchip":
                fn = PchipInterpolator(X[i, k:K], Y[i, k:K], extrapolate=False)

            fn = fn.derivative(num_deriv)
            for j in range(x.size):
                try:
                    y[i, j] = fn(x[j])
                except:
                    # extrapolation was needed (only for UnivariateSpline)
                    pass  # leave as nan
        except:
            # Building the interpolant failed.
            # Not enough valid data points (e.g. X has 1 non-nan value)
            pass  # leave as nan

    # Interpolate with our methods: first, on the fly using ppinterp
    interp_fn = make_pp(interp, out="interp", kind="u")
    y1 = np.empty_like(y)
    for j in range(x.size):
        y1[:, j] = interp_fn(x[j], X, Y, num_deriv)

    # Verify our method is close to SciPy. Won't be bitwise identical.
    assert np.allclose(y, y1, equal_nan=True)

    # Interpolate with our methods:
    # second, with piecewise polynomial coefficients, using ppinterp
    ppc_fn = make_pp(interp, out="coeffs", kind="u")
    Yppc = ppc_fn(X, Y)
    y2 = np.empty_like(y)
    for j in range(x.size):
        y2[:, j] = ppval(x[j], X, Yppc, num_deriv)

    # Our interpolation methods, whether on-the-fly (y1) or building the PPCs
    # for the entire interpolant (y2), should be bitwise identical.
    assert np.array_equal(y1, y2, equal_nan=True)
