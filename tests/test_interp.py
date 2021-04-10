import numpy as np

from neutral_surfaces._interp import linear_coefficients, linear_eval2

# Monotonic but non-uniform pressure grid.
npg = 50
pgrid = np.linspace(2, 1000, npg) ** 1.2

ncasts = 10
S = np.ones((ncasts, npg)) * 35 + np.arange(ncasts)[:, np.newaxis] * 0.01
S *= 0.1 * (pgrid / pgrid[-1]) ** 2
T = np.ones((ncasts, npg)) + np.arange(ncasts)[:, np.newaxis] * 0.01
T *= 10 * (pgrid / pgrid[-1]) ** 3

coefs_T = linear_coefficients(pgrid, T)
coefs_S = linear_coefficients(pgrid, S)


def test_linear():
    fractions = np.linspace(0, 1, 20)
    ntargets = len(fractions)
    targets = pgrid[0] + (pgrid[-1] - pgrid[0]) * fractions
    expected_T = np.empty((ncasts, ntargets), dtype=float)
    expected_S = np.empty_like(expected_T)
    result_S = np.empty_like(expected_T)
    result_T = np.empty_like(expected_T)
    for i in range(ncasts):
        expected_T[i] = np.interp(targets, pgrid, T[i])
        expected_S[i] = np.interp(targets, pgrid, S[i])
        for j in range(ntargets):
            result_S[i, j], result_T[i, j] = linear_eval2(
                targets[j], pgrid, S[i], coefs_S[i], T[i], coefs_T[i]
            )
    assert np.allclose(result_S, expected_S)
    assert np.allclose(result_T, expected_T)
