import pytest
import numpy as np
import numba
from neutral_surfaces._zero import guess_to_bounds, brent


@numba.njit
def univar3(x):
    return (x - 2.5) ** 3 * (x + 2.5) ** 3


@numba.njit
def univar(x):
    return (x - 2.5) * (x + 2.5)


# Test first with empty args.
args = ()


@numba.njit
def one_root():
    a, b = guess_to_bounds(univar, args, -1.0, -6.0, 0.0)
    root = brent(univar, args, a, b, 1e-4)
    return a, b, root


@numba.njit
def one_root3():
    a, b = guess_to_bounds(univar3, args, -1.0, -6.0, 0.0)
    root = brent(univar, args, a, b, 1e-4)
    return a, b, root


@numba.njit
def roots(func, args, tol):
    a, b = guess_to_bounds(func, args, -1.0, -6.0, 0.0)
    root1 = brent(func, args, a, b, tol)
    a, b = guess_to_bounds(func, args, 1.0, 0.0, 6.0)
    root2 = brent(func, args, a, b, tol)
    return root1, root2


@pytest.mark.parametrize("func", [univar, univar3])
def test_root(func):
    tol = 1e-6
    root1, root2 = roots(func, args, tol)
    assert abs(root1 + 2.5) < tol
    assert abs(root2 - 2.5) < tol


# Add an argument.


@numba.njit
def univar_args(x, exp):
    return (x - 2.5) ** exp * (x + 2.5) ** exp


@pytest.mark.parametrize("args", [(1,), (3,)])
def test_root_args(args):
    tol = 1e-6
    root1, root2 = roots(univar_args, args, tol)
    assert abs(root1 + 2.5) < tol
    assert abs(root2 - 2.5) < tol


def test_missing_root():
    tol = 1e-6
    root1, root2 = roots(univar_args, (2,), tol)
    assert np.isnan(root1)
