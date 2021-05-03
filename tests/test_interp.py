import numpy as np

from neutral_surfaces._neutral_surfaces import find_first_nan
from neutral_surfaces.interp_ppc import (
    linear_coeffs,
    pchip_coeffs,
    pchip_coeffs_0d,
    val,
    val_0d,
)
from scipy.interpolate import PchipInterpolator

# Monotonic but non-uniform pressure grid.
npg = 50
P = np.linspace(0, 1000, npg) ** 1.2

ncasts = 3
S = np.empty((ncasts, npg), dtype=np.float64)
S[0] = np.ones(npg) * (P / P[-1]) ** 2  # steadily increasing
S[1] = np.sin(P / P[-1] * 2 * np.pi)  # smooth wave
S[2] = S[1] * np.cos(P / P[-1] * 10 * np.pi)  # crazy wave, with some NaN's
S[2, -3:] = np.nan

# Interpolate to each knot point and each midpoint
p_targets = np.sort(np.concatenate((P, P[0:-1] + np.diff(P) / 2)))


def test_linear():

    Sppc = linear_coeffs(P, S)

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    result_s = np.empty_like(expected_s)

    d_vert = S.ndim - 1  # index to vertical dimension

    assert Sppc.shape[d_vert] == S.shape[d_vert] - 1

    for i in range(ncasts):
        expected_s[i] = np.interp(p_targets, P, S[i])
        for j in range(p_targets.size):
            result_s[i, j] = val(P, S[i], Sppc[i], p_targets[j])
    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_pchip_0d():
    Sppc = pchip_coeffs_0d(P, S[0])

    expected_s = np.empty(p_targets.size, dtype=float)
    result_s = np.empty_like(expected_s)

    d_vert = S[0].ndim - 1  # index to vertical dimension

    assert Sppc.shape[d_vert] == S[0].shape[d_vert] - 1

    k = find_first_nan(S[0])
    SfnP = PchipInterpolator(P[0:k], S[0, 0:k], extrapolate=False)

    for j in range(p_targets.size):
        expected_s[j] = SfnP(p_targets[j])
        result_s[j] = val(P, S[0], Sppc, p_targets[j])
    assert np.allclose(result_s, expected_s, equal_nan=True)


def test_pchip():
    Sppc = pchip_coeffs(P, S)

    expected_s = np.empty((ncasts, p_targets.size), dtype=float)
    result_s = np.empty_like(expected_s)

    d_vert = S.ndim - 1  # index to vertical dimension

    assert Sppc.shape[d_vert] == S.shape[d_vert] - 1

    for i in range(ncasts):
        k = find_first_nan(S[i])
        SfnP = PchipInterpolator(P[0:k], S[i, 0:k], extrapolate=False)

        for j in range(p_targets.size):
            expected_s[i, j] = SfnP(p_targets[j])
            result_s[i, j] = val(P, S[i], Sppc[i], p_targets[j])
    assert np.allclose(result_s, expected_s, equal_nan=True)
