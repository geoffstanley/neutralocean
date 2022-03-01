import numpy as np
import numba as nb
import xarray as xr
import xgcm


def neighbour_rectilinear(dims, conn, periodic):
    """
    Linear indices to each neighbour of each grid point on a regular grid

    This builds a 2D array `adj` giving linear indices to each of `conn`
    neighbours of each grid point in a regular grid.

    Parameters
    ----------
    dims : tuple of int

        The number of grid points in each dimension of the grid.  Currently
        this must be of length 2, i.e. only 2D grids are supported.

    conn : int
        The grid connectivity: for a 2D grid, this must be 4, 5, 8, or 9.

    periodic : tuple of bool
        The periodicity of the grid.  Dimension `i` is periodic iff
        `periodic[i] == True`.


    Returns
    -------
    adj : ndarray

        Linear indices to up to `conn` neighbours of each grid point.
        When a grid point `m` is adjacent to a non-periodic boundary, some of
        `adj[m,:]` will be `N`, where `N = np.prod(dims)` is the total number
        of grid points.

    Notes
    -----
    The connectivity `conn` specifies the number of neighbours to a central
    grid point, possibly including itself.  For `n` between 1 and conn; the
    `n`'th neighbour is located relative to the central grid point according
    to the following diagram:

    conn==4      conn==5    conn==8   conn==9
    +------> j
    | . 1 .       . 1 .     4 1 6     0 3 6
    | 0 . 3       0 2 4     0 . 3     1 4 7
    v . 2 .       . 3 .     5 2 7     2 5 8
    i

    Here, the first dimension (i) increases downward and the second dimension
    (j) increases right.  For example, with `conn == 4`, if `m` is the linear
    index to grid point `(i,j)`, then `n = adj[m,1]` is the linear index to
    grid point `(i-1,j)`.

    """

    ndim = len(dims)  # Number of dimensions in the grid
    assert ndim == 2, "currently only works for 2D grids."
    assert len(periodic) == ndim, "periodic must be a tuple the same length as dims."

    ni = dims[0]
    nj = dims[1]

    badval = ni * nj

    # fmt: off
    # Build adjacency matrix and handle periodicity
    if conn == 4:
        # . 1 .
        # 0 . 3
        # . 2 .
        order = [1, 3, 5, 7]
        adj = _neighbour_rectilinear_helper(ni, nj, order)
        if not periodic[0]:
            adj[0   , :, 1] = badval # i-1 hits a wall when i = 0
            adj[ni-1, :, 2] = badval # i+1 hits a wall when i = ni - 1
        if not periodic[1]:
            adj[:, 0   , 0] = badval # j-1 hits a wall when j = 0
            adj[:, nj-1, 3] = badval # j+1 hits a wall when j = nj - 1

    elif conn == 5:
        # . 1 .
        # 0 4 3
        # . 2 .
        order = [1, 3, 5, 7, 4]
        adj = _neighbour_rectilinear_helper(ni, nj, order)
        if not periodic[0]:
            # adj[0   , :, 1] = badval # i-1 hits a wall when i = 0
            # adj[ni-1, :, 3] = badval # i+1 hits a wall when i = ni - 1
            adj[0   , :, 1] = badval # i-1 hits a wall when i = 0
            adj[ni-1, :, 2] = badval # i+1 hits a wall when i = ni - 1
        if not periodic[1]:
            # adj[:, 0   , 0] = badval # j-1 hits a wall when j = 0
            # adj[:, nj-1, 4] = badval # j+1 hits a wall when j = nj - 1
            adj[:, 0   , 0] = badval # j-1 hits a wall when j = 0
            adj[:, nj-1, 3] = badval # j+1 hits a wall when j = nj - 1
            
        # # . 1 .
        # # 0 2 4
        # # . 3 .
        # # order = [1, 3, 4, 5, 7]
        # adj = _neighbour_rectilinear_helper(ni, nj, order)
        # if not periodic[0]:
        #     adj[0   , :, 1] = badval # i-1 hits a wall when i = 0
        #     adj[ni-1, :, 3] = badval # i+1 hits a wall when i = ni - 1
        # if not periodic[1]:
        #     adj[:, 0   , 0] = badval # j-1 hits a wall when j = 0
        #     adj[:, nj-1, 4] = badval # j+1 hits a wall when j = nj - 1
        #     adj[:, nj-1, 3] = badval # j+1 hits a wall when j = nj - 1
            
    elif conn == 8:
        # 4 1 6
        # 0 . 3
        # 5 2 7
        order = [1, 3, 5, 7, 0, 2, 6, 8]
        adj = _neighbour_rectilinear_helper(ni, nj, order)
        if not periodic[0]:
            adj[0   , :, [1, 4, 6]] = badval # i-1 hits a wall when i = 0
            adj[ni-1, :, [2, 5, 7]] = badval # i+1 hits a wall when i = ni - 1
        if not periodic[1]:
            adj[:, 0   , [0, 4, 5]] = badval # j-1 hits a wall when j = 0
            adj[:, nj-1, [3, 6, 7]] = badval # j+1 hits a wall when j = nj - 1

    elif conn == 9:
        # 0 3 6
        # 1 4 7
        # 2 5 8
        order = range(9)
        adj = _neighbour_rectilinear_helper(ni, nj, order)
        if not periodic[0]:
            adj[0   , :, [0, 3, 6]] = badval # i-1 hits a wall when i = 0
            adj[ni-1, :, [2, 5, 8]] = badval # i+1 hits a wall when i = ni - 1
        if not periodic[1]:
            adj[:, 0   , [0, 1, 2]] = badval # j-1 hits a wall when j = 0
            adj[:, nj-1, [6, 7, 8]] = badval # j+1 hits a wall when j = nj - 1

    else:
        raise("Unknown number of neighbours.  conn must be one of 4, 5, 8, or 9.")
    # fmt: on

    # Reshape adj to a matrix of dimensions (conn, ni*nj)
    adj = adj.reshape((ni * nj, conn))

    return adj


def _neighbour_rectilinear_helper(ni, nj, order):

    D = len(order)  # max degree

    # Prepare to circshift linear indices to some subset of its neighbours
    # generally ordered as follows
    # +------> j = 2'nd dim
    # | 0 3 6
    # | 1 4 7
    # | 2 5 8
    # v
    #  i = 1'st dim
    spin = (
        (1, 1),
        (0, 1),
        (-1, 1),
        (1, 0),
        (0, 0),
        (-1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )  # - sign included as prep for np.roll

    # Build linear index to each grid point, and repeat them D times
    adj = np.tile(np.reshape(range(ni * nj), (ni, nj, 1)), (1, 1, D))  # ni x nj x D

    # Shift these linear indices so they refer to their neighbours.
    for d in range(D):
        adj[:, :, d] = np.roll(adj[:, :, d], spin[order[d]], (0, 1))

    return adj


@nb.njit
def find_first(item, vec):
    """return the index of the first occurence of `item` in `vec`
    If not found, return -1."""
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
# In the xgcm language, that is ordered as
# Yleft, Xleft, Xright, Yright
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


def edgescompact_from_faceconns(face_connections, n):
    # Access first (and only) key / value of face_connections
    fname = next(iter(face_connections.keys()))  # name of face_connections only key
    nf = len(next(iter(face_connections.values())))  # len(faces_connections only value)

    gg = xr.Dataset(None, {"i": np.arange(n), "j": np.arange(n), fname: np.arange(nf)})
    grid = xgcm.Grid(
        gg,
        periodic=False,
        face_connections=face_connections,
        coords={
            # "X": {"center": "i", "left": "i", "right": "i"},
            # "Y": {"center": "j", "left": "j", "right": "j"},
            "X": {"center": "i", "left": "i"},
            "Y": {"center": "j", "left": "j"},
        },
    )

    # Make a DataArray but filled with 0, 1, ...
    idx = xr.DataArray(
        np.arange(nf * n * n).reshape((nf, n, n)),
        dims=(fname, "j", "i"),
        coords=(np.arange(nf), np.arange(n), np.arange(n)),
    )

    # Note j is the Y axis and i is the X axis, when we write data[f, j, i]  (f the face index)
    idx_im1 = grid.axes["X"]._neighbor_binary_func(
        idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
    )
    idx_jm1 = grid.axes["Y"]._neighbor_binary_func(
        idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
    )

    # idx_ip1 = grid.axes["X"]._neighbor_binary_func(
    #     idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
    # )
    # idx_jp1 = grid.axes["Y"]._neighbor_binary_func(
    #     idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
    # )

    # adj = np.empty((n * n * nf, 4), dtype=int)
    # adj = np.empty((n * n * nf, 2), dtype=int)
    # adj[:, 0] = idx_jm1.values.reshape(-1)
    # adj[:, 1] = idx_im1.values.reshape(-1)
    # adj[:, 2] = idx_ip1.values.reshape(-1)
    # adj[:, 3] = idx_jp1.values.reshape(-1)
    # return adj

    return np.stack((idx_im1.values.reshape(-1), idx_jm1.values.reshape(-1)), axis=-1)


# Equivalent but slower than below nb.njit'ed function.
# def adj_to_edges(adj):
#     N, d = adj.shape
#     return np.stack((np.tile(np.arange(N), d), adj.T.reshape(-1)), axis=1)


@nb.njit
def adj_to_edges(adj):
    N, D = adj.shape
    edges = np.empty((N * D, 2), dtype=type(0))
    for d in range(D):
        for n in range(N):
            # edges[n * D + d, 0] = n
            # edges[n * D + d, 1] = adj[n,d]
            edges[d * N + n, :] = (n, adj[n, d])
    return edges
