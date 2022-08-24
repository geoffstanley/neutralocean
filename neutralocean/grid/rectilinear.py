import numpy as np


def build_edges(dims, periodic):
    """
    Build list of pairs of adjacent grid points, numbered 1, ..., N, on a
    rectilinear grid

    Parameters
    ----------
    dims : tuple of int
        dimensions of the grid.  i'th element gives number of grid cells in the
        i'th direction, for i = 1,2.
    periodic : tuple of bool
        Specifies periodicity.  The i'th dimension is periodic when `periodic[i]` is True.

    Returns
    -------
    edges : array
        List of pairs of adjacent grid points.
        If `dims == (ni,nj)`, then the grid points are labelled 1, 2, ..., N=ni*nj.
        The grid points whose integer indices are `edges[i,0]` and `edges[i,1]`
        are adjacent.
        (Pairs that are adjacent in the first dimension come first in `edges`,
         followed by pairs that are adjacent in the second dimension.)
        If both elements of `periodic` are True, there will be 2 * ni * nj pairs
        of adjacent grid points (rows of `edges`).  Otherwise, there will be fewer.

    """

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
    Build a 1D array of `data` in the same order as the `edges` constructed by `build_edges`

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
        1D array of `data`.
        If `edges = build_edges(dims, periodic)`, then `edge_data[i]` is the
        value from `data` that corresponds to the interface between the grid
        cells indexed by `edges[i,0]` and `edges[i,1]`.

    """
    i1 = int(not periodic[0])  # 0 when periodic, 1 when non-periodic
    j1 = int(not periodic[1])
    x = np.broadcast_to(data[0], dims)
    y = np.broadcast_to(data[1], dims)
    return np.concatenate((x[i1:, :].reshape(-1), y[:, j1:].reshape(-1)))
