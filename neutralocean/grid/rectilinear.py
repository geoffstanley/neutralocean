import numpy as np


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


def build_edges(dims, periodic):
    ndim = len(dims)  # Number of dimensions in the grid
    assert ndim == 2, "currently only works for 2D grids."
    assert len(periodic) == ndim, "periodic must be a tuple the same length as dims."

    ni, nj = dims
    N = ni * nj  # number of nodes

    i1 = int(not periodic[0])  # 0 when periodic, 1 when non-periodic
    j1 = int(not periodic[1])
    i2 = ni - i1  # ni when periodic, ni - 1 when non-periodic
    j2 = nj - j1

    # Calculate number of edges
    Ei = i2 * nj
    Ej = ni * j2
    E = Ei + Ej

    edges = np.empty((E, 2), dtype=int)  # prealloc space

    # Build linear indices to the entire grid
    idx = np.arange(N).reshape((ni, nj))

    # Handle first dimension
    if periodic[0]:
        edges[:Ei, 0] = idx.reshape(-1)
        edges[:Ei, 1] = np.roll(idx, 1, axis=0).reshape(-1)
    else:
        edges[:Ei, 0] = idx[i1:, :].reshape(-1)
        edges[:Ei, 1] = idx[:i2, :].reshape(-1)

    # Handle second dimension
    if periodic[1]:
        edges[Ei:, 0] = idx.reshape(-1)
        edges[Ei:, 1] = np.roll(idx, 1, axis=1).reshape(-1)
    else:
        edges[Ei:, 0] = idx[:, j1:].reshape(-1)
        edges[Ei:, 1] = idx[:, :j2].reshape(-1)

    return edges


def build_edge_data(dims, periodic, data):
    """


    Parameters
    ----------
    dims : tuple of int
        dimensions of the grid.  i'th element gives number of grid cells in the
        i'th direction, for i = 1,2.
    periodic : tuple of bool
        Specifies periodicity.  The i'th dimension is periodic when `periodic[i]` is True.
    data : tuple of float or ndarray
        The data that lives on edges, i.e. between water columns.
        The i'th element gives data that lives between water columns in the i'th
        dimension.
        For example, `data = (xdist, ydist)` where `xdist` is the distance
        between water columns in the x (first) dimension, and `ydist` is the
        distance between water columns in the y (second) dimension.

    Returns
    -------
    edge_data : array
        1D array of `data` in the same order as the `edges` constructed by
        `build_edges`.

    """
    i1 = int(not periodic[0])  # 0 when periodic, 1 when non-periodic
    j1 = int(not periodic[1])
    x = np.broadcast_to(data[0], dims)
    y = np.broadcast_to(data[1], dims)
    return np.concatenate((x[i1:, :].reshape(-1), y[:, j1:].reshape(-1)))
