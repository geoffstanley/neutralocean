import numpy as np
import numba as nb


@nb.njit
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


# Index a "2D" data field, e.g. sea-surface temperature, as [f, j, i]
# where f is the face or tile index, j is usually the "y" index, and i is
# usually the "x" index.
#
#   +------> i
#   | . 0 .
#   | 1 . 2
#   | . 3 .
# j v
# That is,
# 0 indexes (j-1, i)
# 1 indexes (j, i-1)
# 2 indexes (j, i+1)
# 3 indexes (j+1, i)
def neighbour4_tiled_rectilinear(F, n):
    """
    Make linear indices to the 4 neighbours of point on a tiled rectilinear grid

    Parameters
    ----------

    F : ndarray of int

    n : int
        Number of grid points on each side of a square tile.
        (Note this code cannot handle non-square tiles.)

    Return
    ------
    A : ndarray of int


    """

    nf = F.shape[0]  # number of faces

    n2 = n * n

    a = np.arange(n)
    f_ = np.arange(nf).reshape((-1, 1, 1))
    j_ = a.reshape((1, -1, 1))
    i_ = a.reshape((1, 1, -1))

    A = np.empty((nf * n2, 4), dtype=int)

    # Make linear index to (f, j-1, i)
    idx = f_ * n2 + (j_ - 1) * n + i_
    for f in range(nf):
        f2 = F[f, 0]
        if f2 == -1:
            idx[f, 0, :] = -1
        else:
            side = find_first(f, F[f2, :])
            if side == 0:
                # link to face f2 on j==0 side: (f, 0, i) -> (f2, 0, n-1-i)
                idx[f, 0, :] = f2 * n2 + n - 1 - a
            elif side == 1:
                # link to face f2 on i==0 side: (f, 0, i) -> (f2, i, 0)
                idx[f, 0, :] = f2 * n2 + a * n
            elif side == 2:
                # link to face f2 on i==n-1 side: (f, 0, i) -> (f2, n-1-i, n-1)
                idx[f, 0, :] = f2 * n2 + (n - 1 - a) * n + (n - 1)
            elif side == 3:
                # link to face f2 on j==n-1 side: (f, 0, i) -> (f2, n-1, i)
                idx[f, 0, :] = f2 * n2 + (n - 1) * n + a
    A[:, 0] = idx.reshape(-1)

    # Make linear index to (f, j, i-1)
    idx = f_ * n2 + j_ * n + i_ - 1
    for f in range(nf):
        f2 = F[f, 1]
        if f2 == -1:
            idx[f, :, 0] = -1
        else:
            side = find_first(f, F[f2, :])
            if side == 0:
                # link to face f2 on j==0 side: (f, j, 0) -> (f2, 0, i)
                idx[f, 0, :] = f2 * n2 + a
            elif side == 1:
                # link to face f2 on i==0 side: (f, j, 0) -> (f2, n-1-j, 0)
                idx[f, 0, :] = f2 * n2 + (n - 1 - a) * n
            elif side == 2:
                # link to face f2 on i==n-1 side: (f, j, 0) -> (f2, j, n-1)
                idx[f, 0, :] = f2 * n2 + a * n + n - 1
            elif side == 3:
                # link to face f2 on j==n-1 side: (f, j, 0) -> (f2, n-1, n-1-j)
                idx[f, 0, :] = f2 * n2 + (n - 1) * n + n - 1 - a
            # else:
            #    raise RuntimeError("unexpected face connection")
    A[:, 1] = idx.reshape(-1)

    # Make linear index to (f, j, i+1)
    idx = f_ * n2 + j_ * n + i_ + 1
    for f in range(nf):
        f2 = F[f, 2]
        if f2 == -1:
            idx[f, :, -1] = -1
        else:
            side = find_first(f, F[f2, :])
            if side == 0:
                # link to face f2 on j==0 side: (f, j, n-1) -> (f2, 0, n-1-j)
                idx[f, :, -1] = f2 * n2 + n - 1 - a
            elif side == 1:
                # link to face f2 on i==0 side: (f, j, n-1) -> (f2, j, 0)
                idx[f, :, -1] = f2 * n2 + a * n
            elif side == 2:
                # link to face f2 on i==n-1 side: (f, j, n-1) -> (f2, n-1-j, n-1)
                idx[f, :, -1] = f2 * n2 + (n - 1 - a) * n + (n - 1)
            elif side == 3:
                # link to face f2 on j==n-1 side: (f, j, n-1) -> (f2, n-1, j)
                idx[f, :, -1] = f2 * n2 + (n - 1) * n + a
    A[:, 2] = idx.reshape(-1)

    # Make linear index to (f, j+1, i)
    idx = f_ * n2 + (j_ + 1) * n + i_
    for f in range(nf):
        f2 = F[f, 3]
        if f2 == -1:
            idx[f, -1, :] = -1
        else:
            side = find_first(f, F[f2, :])
            if side == 0:
                # link to face f2 on j==0 side: (f, n-1, i) -> (f2, 0, i)
                idx[f, -1, :] = f2 * n2 + a
            elif side == 1:
                # link to face f2 on i==0 side: (f, n-1, i) -> (f2, n-1-i, 0)
                idx[f, -1, :] = f2 * n2 + (n - 1 - a) * n
            elif side == 2:
                # link to face f2 on i==n-1 side: (f, n-1, i) -> (f2, i, n-1)
                idx[f, -1, :] = f2 * n2 + a * n + (n - 1)
            elif side == 3:
                # link to face f2 on j==n-1 side: (f, n-1, i) -> (f2, n-1, n-1-i)
                idx[f, -1, :] = f2 * n2 + (n - 1) * n + (n - 1 - a)
    A[:, 3] = idx.reshape(-1)

    return A


# In the xgcm language, that is ordered as
# Yleft, Xleft, Xright, Yright
def xgcm_faceconns_convert(face_connections):
    fc = tuple(face_connections.values())[0]
    nf = len(tuple(fc.keys()))

    F = np.empty((nf, 4), dtype=int)
    for i in range(nf):
        a = fc[i]["Y"][0]  # Yleft
        F[i, 0] = -1 if a is None else a[0]

        a = fc[i]["X"][0]  # Xleft
        F[i, 1] = -1 if a is None else a[0]

        a = fc[i]["X"][1]  # Xright
        F[i, 2] = -1 if a is None else a[0]

        a = fc[i]["Y"][1]  # Yright
        F[i, 3] = -1 if a is None else a[0]

    return F
