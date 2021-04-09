import numpy as np
import numba

def linear_coefficients(p, y):
    return np.diff(y, axis=-1) / np.diff(p, axis=-1)

@numba.njit
def search(p, ptarget):
    for ii in range(len(p)):
        if p[ii] > ptarget:
            return ii - 1
    raise RuntimeError("failed to find target")  # can't include the value...


@numba.njit
def linear_eval2(ptarget, p, y1, c1, y2, c2):
    ii = search(p, ptarget)
    dp = p[ii] - ptarget
    y_int1 = y1[ii] + dp * c1[ii]
    y_int2 = y2[ii] + dp * c2[ii]
    return (y_int1, y_int2)
