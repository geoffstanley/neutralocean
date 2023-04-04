import numpy as np


def build_grid(dims, periodic, dxC=1.0, dyC=1.0, dyG=1.0, dxG=1.0):
    """
    Build pairs of adjacent grid points and associated geometry for a rectilinear grid

    Parameters
    ----------
    dims : tuple of int
        dimensions of the grid.  i'th element gives number of grid cells in the
        i'th direction, for i = 1,2.

    periodic : tuple of bool
        Specifies periodicity.  The i'th dimension is periodic when `periodic[i]` is True.

    dxC : float or ndarray, Default 1.0
        Distance between adjacent grid points in the 1st ('x') dimension

    dyC : float or ndarray, Default 1.0
        Distance between adjacent grid points in the 2nd ('y') dimension

    dyG : float or ndarray, Default 1.0
        Distance (in the 2nd, 'y' dimension) of the face between grid points
        that are adjacent in the 1st ('x') dimension
        Lives at same location as `dxC`.

    dxG : float or ndarray, Default 1.0
        Distance (in the 1st, 'x' dimension) of the face between grid points
        that are adjacent in the 2nd ('y') dimension.
        Lives at same location as `dyC`.

    Returns
    -------
    grid : dict
        Containing the following:

        edges : tuple of length 2
            Each element is an array of int of length E, where E is the number of
            edges in the grid's graph, i.e. the number of pairs of adjacent water
            columns (including land) in the grid.
            If `edges = (a, b)`, the nodes (water columns) whose linear indices are
            `a[i]` and `b[i]` are adjacent.

        dist : 1d array
            Horizontal distance between adjacent water columns (nodes).
            `dist[i]` is the distance between nodes whose linear indices are
            `edges[0][i]` and `edges[1][i]`.

        distperp : 1d array
            Horizontal distance of the face between adjacent water columns (nodes).
            `distperp[i]` is the distance of the interface between nodes whose
            linear indices are `edges[0][i]` and `edges[1][i]`.
    """
    grid = dict()
    grid["edges"] = _build_edges(dims, periodic)
    grid["dist"] = _build_edgedata(dims, periodic, (dxC, dyC))
    grid["distperp"] = _build_edgedata(dims, periodic, (dyG, dxG))
    return grid


def _build_edges(dims, periodic):
    """
    Build list of pairs of adjacent grid points, numbered 0, ..., N-1, on a
    rectilinear grid
    """

    ndim = len(dims)  # Number of dimensions in the grid
    assert ndim == 2, "currently only works for 2D grids."
    assert (
        len(periodic) == ndim
    ), "periodic must be a tuple the same length as dims."

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

    a, b = (np.empty(E, dtype=int) for x in (0, 1))  # prealloc space

    # Build linear indices to the entire grid
    idx = np.arange(N).reshape((ni, nj))

    # Handle first dimension
    if periodic[0]:
        a[:Ei] = idx.reshape(-1)
        b[:Ei] = np.roll(idx, 1, axis=0).reshape(-1)
    else:
        a[:Ei] = idx[i1:, :].reshape(-1)
        b[:Ei] = idx[:i2, :].reshape(-1)

    # Handle second dimension
    if periodic[1]:
        a[Ei:] = idx.reshape(-1)
        b[Ei:] = np.roll(idx, 1, axis=1).reshape(-1)
    else:
        a[Ei:] = idx[:, j1:].reshape(-1)
        b[Ei:] = idx[:, :j2].reshape(-1)

    return a, b


def _build_edgedata(dims, periodic, data):
    """
    Build a 1D array of `data` in the same order as constructed by `_build_edges`

    Parameters
    ----------
    dims, periodic :
        See `build_grid`

    data : tuple of ndarray
        The data that lives on edges, i.e. between water columns.
        The i'th element of the tuple gives data that lives between water
        columns in the i'th dimension.
        Will be broadcast to the size `dims`.
        Example: `data = (xdist, ydist)` where `xdist` is the distance
        between water columns in the x (first) dimension, and `ydist` is the
        distance between water columns in the y (second) dimension.

    Returns
    -------
    edgedata : array
        1D array of `data`.
        If `a, b = _build_edges(dims, periodic)`, then `edgedata[i]` is the
        value from `data` that corresponds to the interface between the grid
        cells indexed by `a[i]` and `b[i]`.

    """
    i1 = int(not periodic[0])  # 0 when periodic, 1 when non-periodic
    j1 = int(not periodic[1])
    x = np.broadcast_to(data[0], dims)
    y = np.broadcast_to(data[1], dims)
    return np.concatenate((x[i1:, :].reshape(-1), y[:, j1:].reshape(-1)))


def edgedata_to_maps(edgedata, dims, periodic):
    """
    Convert 1D array of data living on edges into two 2D arrays, one for each
    spatial dimension, for a rectilinear grid

    Parameters
    ----------
    edgedata : array
        1D array of data that lives on edges of the grid's graph, given in the
        same order as the edges cosntructed from `_build_edges`.
        If `a, b = _build_edges(dims, periodic)`, then `edgedata[i]` is the
        data that lives at the interface between the grid cells indexed by
        `a[i]` and `b[i]`.

    dims, periodic :
        See `build_grid`

    Returns
    -------

    Fi, Fj : ndarray
        2D arrays of shape `dims`.  `Fi` contains the values from `edgedata`
        that correspond to data living between grid points in the first
        dimension, and similarly for `Fj` but in the second dimension.
    """

    ni, nj = dims
    N = ni * nj  # number of nodes

    i1 = int(not periodic[0])  # 0 when periodic, 1 when non-periodic
    j1 = int(not periodic[1])

    Fi, Fj = (np.full((ni, nj), np.nan) for x in (0, 1))

    Fi[i1:, :] = edgedata[:N].reshape((ni - i1, nj))
    Fj[:, j1:] = edgedata[N:].reshape((ni, nj - j1))

    return Fi, Fj
