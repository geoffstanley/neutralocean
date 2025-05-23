import pytest
import numpy as np
import numba
from neutralocean.fzero import brent_guess

tol = 1e-6

# Test with univariate function
@numba.njit
def univar(x):
    return x - 2.5


@numba.njit
def univar3(x):
    return (x - 2.5) ** 3


@pytest.mark.parametrize("func", [univar, univar3])
def test_root(func):
    root = brent_guess(func, 1.0, 0.0, 6.0, tol)
    assert abs(root - 2.5) < tol


# Test when one end of the search range is a root
def test_ub_eq_root():
    root = brent_guess(univar, 1.0, 0.0, 2.5, tol)
    assert abs(root - 2.5) < tol


def test_lb_eq_root():
    root = brent_guess(univar, 4.0, 2.5, 6.0, tol)
    assert abs(root - 2.5) < tol


# Test function with extra parameters
@numba.njit
def univar_args(x, exp):
    return (x - 2.5) ** exp * (x + 2.5) ** exp


@pytest.mark.parametrize("args", [(1,), (3,)])
def test_root_args(args):
    root = brent_guess(univar_args, 1.0, 0.0, 6.0, tol, args)
    assert abs(root - 2.5) < tol


# Test function which does not change sign at its roots, i.e. a quadratic
def test_root_singular():
    root = brent_guess(univar_args, 1.0, 0.0, 6.0, tol, (2,))
    assert np.isnan(root)


# Test function which does not change sign at its roots, but which is zero at
# one end of the search range
def test_ub_eq_root_singular():
    root = brent_guess(univar_args, 1.0, 0.0, 2.5, tol, (2,))
    assert abs(root - 2.5) < tol


def test_lb_eq_root_singular():
    root = brent_guess(univar_args, 4.0, 2.5, 6.0, tol, (2,))
    assert abs(root - 2.5) < tol
