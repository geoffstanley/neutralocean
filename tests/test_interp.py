import numpy as np

from neutral_surfaces.interp_ppc import linear_coefficients, val2

# Monotonic but non-uniform pressure grid.
npg = 50
pgrid = np.linspace(2, 1000, npg) ** 1.2

ncasts = 10
S = np.ones((ncasts, npg)) * 35 + np.arange(ncasts)[:, np.newaxis] * 0.01
S *= 0.1 * (pgrid / pgrid[-1]) ** 2
T = np.ones((ncasts, npg)) + np.arange(ncasts)[:, np.newaxis] * 0.01
T *= 10 * (pgrid / pgrid[-1]) ** 3

Tppc = linear_coefficients(pgrid, T)
Sppc = linear_coefficients(pgrid, S)


def test_linear():
    fractions = np.linspace(0, 1, 20)
    ntargets = len(fractions)
    targets = pgrid[0] + (pgrid[-1] - pgrid[0]) * fractions
    expected_T = np.empty((ncasts, ntargets), dtype=float)
    expected_S = np.empty_like(expected_T)
    result_S = np.empty_like(expected_T)
    result_T = np.empty_like(expected_T)

    d_vert = S.ndim - 1  # index to vertical dimension

    assert Sppc.shape[d_vert] == S.shape[d_vert] - 1
    assert Tppc.shape[d_vert] == T.shape[d_vert] - 1

    for i in range(ncasts):
        expected_T[i] = np.interp(targets, pgrid, T[i])
        expected_S[i] = np.interp(targets, pgrid, S[i])
        for j in range(ntargets):
            result_S[i, j], result_T[i, j] = val2(
                pgrid, S[i], Sppc[i], T[i], Tppc[i], targets[j]
            )
    assert np.allclose(result_S, expected_S)
    assert np.allclose(result_T, expected_T)
