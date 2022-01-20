import numpy as np
import pytest

from neutralocean.lib import find_first_nan
from neutralocean.interp import (
    interp,
    linterp_i,
    linterp_dx_i,
    pchip_i,
    pchip_dx_i,
    pchip_dxx_i,
)
from scipy.interpolate import UnivariateSpline, PchipInterpolator

N = 3  # number of 1D interpolation problems
K = 10  # number of grid points in each interpolation problem

# Monotonic but non-uniform independent data
X1 = np.linspace(0, 10, K) ** 1.2
X = np.tile(X1, (N, 1))

# Build dependent data
Y = np.empty((N, K), dtype=np.float64)
Y[0] = np.ones(K) * (X1 / X1[-1]) ** 2  # steadily increasing
Y[1] = np.sin(X1 / X1[-1] * 2 * np.pi)  # smooth wave
Y[2] = Y[1] * np.cos(X1 / X1[-1] * 10 * np.pi)  # crazy wave, with some NaN's
Y[2, -3:] = np.nan


# X[0,1:] = np.nan   # need to test this kind of case out!


# Interpolate between each knot
x_midpts = X1[0:-1] + np.diff(X1) / 2

# Interpolate to each knot point and each midpoint
x_targets = np.sort(np.concatenate((X1, x_midpts)))


# %%

# Note re: lhs vs rhs:
# Our code won't agree with SciPy when evaluating a function that is piecewise
# discontinuous (e.g. first derivative of a linear interpolant or second
# derivative of a PHCIP) at the knots, because we evaluate using the "left"
# side whereas SciPy evaluates using the "right" side.
@pytest.mark.parametrize(
    "interp_fn,num_deriv,x",
    [
        (linterp_i, 0, x_targets),
        (linterp_dx_i, 1, x_midpts),  # see "Note re: lhs vs rhs"
        (pchip_i, 0, x_targets),
        (pchip_dx_i, 1, x_targets),
        (pchip_dxx_i, 2, x_midpts),  # see "Note re: lhs vs rhs"
    ],
)
def test_interp(interp_fn, num_deriv, x):

    # Interpolate with SciPy
    y = np.empty((N, x.size), dtype=float)
    for i in range(N):
        k = find_first_nan(Y[i])
        if interp_fn.__name__.startswith("linterp"):
            fn = UnivariateSpline(X[i, 0:k], Y[i, 0:k], k=1, s=0, ext="raise")
        elif interp_fn.__name__.startswith("pchip"):
            fn = PchipInterpolator(X[i, 0:k], Y[i, 0:k], extrapolate=False)
        fn = fn.derivative(num_deriv)
        for j in range(x.size):
            try:
                y[i, j] = fn(x[j])
            except:
                # extrapolation was needed (only for UnivariateSpline)
                y[i, j] = np.nan

    # Interpolate with our methods
    y_ = np.empty_like(y)
    for j in range(x.size):
        y_[:, j] = interp(x[j], X, Y, interp_fn)

    assert np.allclose(y, y_, equal_nan=True)
