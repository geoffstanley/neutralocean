import pytest

import numba
from neutral_surfaces._zero import guess_to_bounds, brent


@numba.njit
def univar3(x):
    return (x - 2.5) ** 3 * (x + 2.5) ** 3


@numba.njit
def univar(x):
    return (x - 2.5) * (x + 2.5)


@numba.njit
def one_root():
    a, b = guess_to_bounds(univar, -1.0, -6.0, 0.0)
    root = brent(univar, a, b, 1e-4)
    return a, b, root


@numba.njit
def one_root3():
    a, b = guess_to_bounds(univar3, -1.0, -6.0, 0.0)
    root = brent(univar, a, b, 1e-4)
    return a, b, root


@numba.njit
def roots(func, tol):
    a, b = guess_to_bounds(func, -1.0, -6.0, 0.0)
    root1 = brent(func, a, b, tol)
    a, b = guess_to_bounds(func, 1.0, 0.0, 6.0)
    root2 = brent(func, a, b, tol)
    return root1, root2


@pytest.mark.parametrize("func", [univar, univar3])
def test_root(func):
    tol = 1e-6
    root1, root2 = roots(func, tol)
    assert abs(root1 + 2.5) < tol
    assert abs(root2 - 2.5) < tol
