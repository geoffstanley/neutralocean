import numpy as np

from neutral_surfaces._neutral_surfaces import find_first_nan
from neutral_surfaces.interp_ppc import (
    linear_coeffs,
    pchip_coeffs,
    val,
    deriv,
)
from scipy.interpolate import PchipInterpolator

# Monotonic but non-uniform pressure grid.
npg = 10
P = np.linspace(0, 1000, npg) ** 1.2

ncasts = 3
S = np.empty((ncasts, npg), dtype=np.float64)
S[0] = np.ones(npg) * (P / P[-1]) ** 2  # steadily increasing
S[1] = np.sin(P / P[-1] * 2 * np.pi)  # smooth wave
S[2] = S[1] * np.cos(P / P[-1] * 10 * np.pi)  # crazy wave, with some NaN's
S[2, -3:] = np.nan


# Interpolate between each point, but do not include the knots
p_midpts = P[0:-1] + np.diff(P) / 2

# Interpolate to each knot point and each midpoint
p_targets = np.sort(np.concatenate((P, p_midpts)))


def test_linear():

    Sppc = linear_coeffs(P, S)

    d_vert = S.ndim - 1  # index to vertical dimension
    assert Sppc.shape[d_vert] == S.shape[d_vert] - 1

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    for i in range(ncasts):
        expected_s[i] = np.interp(p_targets, P, S[i])

    result_s = np.empty_like(expected_s)
    for j in range(p_targets.size):
        result_s[:, j] = val(P, S, Sppc, p_targets[j])

    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_pchip():
    Sppc = pchip_coeffs(P, S)

    d_vert = S.ndim - 1  # index to vertical dimension
    assert Sppc.shape[d_vert] == S.shape[d_vert] - 1

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    for i in range(ncasts):
        k = find_first_nan(S[i])
        SfnP = PchipInterpolator(P[0:k], S[i, 0:k], extrapolate=False)
        for j in range(p_targets.size):
            expected_s[i, j] = SfnP(p_targets[j])

    result_s = np.empty_like(expected_s)
    for j in range(p_targets.size):
        result_s[:, j] = val(P, S, Sppc, p_targets[j])

    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_deriv1():
    num_deriv = 1
    Sppc = pchip_coeffs(P, S)

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    for i in range(ncasts):
        k = find_first_nan(S[i])
        SPfnP = PchipInterpolator(P[0:k], S[i, 0:k], extrapolate=False).derivative(
            num_deriv
        )
        for j in range(p_targets.size):
            expected_s[i, j] = SPfnP(p_targets[j])

    result_s = np.empty_like(expected_s)
    for j in range(p_targets.size):
        result_s[:, j] = val(P, S, Sppc, p_targets[j], num_deriv)

    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_deriv2():
    # Note, our code won't agree with SciPy when evaluating the second
    # derivative of a PHCIP at the knots, because this second derivative is
    # discontinuous and we evaluate using the "left" side whereas SciPy
    # evaluates using the "right" side.
    num_deriv = 2
    Sppc = pchip_coeffs(P, S)

    expected_s = np.empty((ncasts, p_midpts.size), dtype=float)
    for i in range(ncasts):
        k = find_first_nan(S[i])
        SPfnP = PchipInterpolator(P[0:k], S[i, 0:k], extrapolate=False).derivative(
            num_deriv
        )

        for j in range(p_midpts.size):
            expected_s[i, j] = SPfnP(p_midpts[j])

    result_s = np.empty_like(expected_s)
    for j in range(p_midpts.size):
        result_s[:, j] = val(P, S, Sppc, p_midpts[j], num_deriv)

    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_ppc_deriv():
    num_deriv = 2
    Sppc = pchip_coeffs(P, S)
    dS, dSppc = deriv(P, S, Sppc, num_deriv)

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    result_s = np.empty_like(expected_s)
    for j in range(p_targets.size):
        expected_s[:, j] = val(P, S, Sppc, p_targets[j], num_deriv)
        result_s[:, j] = val(P, dS, dSppc, p_targets[j])

    assert np.allclose(result_s, expected_s, equal_nan=True)
