import numpy as np


def build_grid(dims, e1u=1.0, e2v=1.0, e2u=1.0, e1v=1.0):
    """
    Build pairs of adjacent grid points and associated geometry for a tripolar grid

    Parameters
    ----------
    dims : tuple of int
        Dimensions of the grid. Letting `dims == (nj, ni)`, then
        `nj` is the number of grid cells in the y (latitude-like) direction, &
        `ni` is the number of grid cells in the x (longitude-like) direction.

    e1u : float or 2D array, Default 1.0
        Distance in the x dimension that lives on the U cell.

    e2v : float or 2D array, Default 1.0
        Distance in the y dimension that lives on the V cell.

    e2u : float or 2D array, Default 1.0
        Distance in the y dimension that lives on the U cell.

    e1v : float or 2D array, Default 1.0
        Distance in the x dimension that lives on the V cell.

    Notes
    -----
    `e1u`, `e2v`, `e2u`, `e1v` can have shape `(nj,ni)` or `(nj+1, ni+2)`.
    The latter form has 1 cell of padding in the y dimension to handle the
    tripolar grid, and 1 cell of padding in the x dimension to handle the
    longitudinal periodicity. If given the latter form, this extra padding is
    removed internally. When of shape `(nj, ni)`, 

    - `e1u[j,i]` is the x distance on the U grid at [j,i+1/2],
    - `e2v[j,i]` is the y distance on the V grid at [j+1/2,i],
    - `e2u[j,i]` is the y distance on the U grid at [j,i+1/2],
    - `e1v[j,i]` is the x distance on the V grid at [j+1/2,i].

    Returns
    -------
    grid : dict
        Containing the following:

        edges : tuple of length 2
            Each element is an array of int of length `E`, where `E` is the number of
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
    grid["edges"] = _build_edges(dims)
    grid["dist"] = _build_edgedata(dims, (e2v, e1u))
    grid["distperp"] = _build_edgedata(dims, (e1v, e2u))
    return grid


def _build_edges(dims):
    """
    Build list of pairs of adjacent grid points, numbered 0, ..., N-1, on a
    tripolar grid (for ORCA)
    """

    assert len(dims) == 2, "Need `len(dims) == 2`, for a 2D horizontal grid."

    nj, ni = dims

    Ei = ni * nj  # number of edges in i dimension
    Ej = ni * nj  # number of edges in j dimension
    E = Ei + Ej  #  number of edges in total

    a = np.empty(E, dtype=int)  # prealloc space
    b = np.empty(E, dtype=int)

    # Build linear indices to the entire grid with one extra row of padding
    # for the northern latitudes. Do not add longitudinal padding.
    idx = np.arange((nj + 1) * ni).reshape((nj + 1, ni))

    # Handle tripolar grid's northern latitudes: using the extra row of padding,
    # the cell indexed (here using 0-indexing) by [nj - 1, i] is actually in the
    # domain (not the padded row) and is adjacent to [nj - 1, ni - i - 1]. So,
    # make the linear index at cell [nj, i] have the same value as the linear
    # index at cell [nj - 1, ni - i - 1].
    idx[-1, :] = np.flip(idx[-2, :])

    # Handle j dimension (non-periodic).
    # Cells [j, i] and [j+1, i] are adjacent, for 0 <= j <= nj-1. This is true
    # even for j == nj-1, thanks to the extra padded row and np.flip above.
    a[:Ej] = idx[:nj, :].reshape(-1)  # [j, i]
    b[:Ej] = idx[1:, :].reshape(-1)  # [j+1, i]

    # Handle i dimension (zonally periodic).
    # Cells [j, i] and [j, mod(i+1, ni)] are adjacent.
    a[Ej:] = idx[0:-1, :].reshape(-1)  # [j, i]
    b[Ej:] = np.roll(idx[0:-1, :], -1, axis=1).reshape(-1)  # [j, i+1]

    return a, b


def _build_edgedata(dims, data):
    """
    Build a 1D array of `data` in the same order as constructed by `_build_edges`

    Parameters
    ----------
    dims :
        See `build_grid`

    data : tuple of ndarray
        Data that lives on edges, i.e. between nodes (water columns).

        With `dims == (nj, ni)`, each element of `data` should be a 2D array of
        shape `(nj, ni)` --- but if they are of shape `(nj+1, ni+2)` they will
        be trimmed to remove the last row (duplication of the north fold) and 
        the first and last columns (duplication for zonal periodicity).

        - `data[0][j, i]` lives between nodes [i, j] and [i, j+1], for `0 <= j <= nj-1`.

        - `data[1][j, i]` lives between nodes [i, j] and [i+1, j], for `0 <= i <= ni-1`.

        Example: `data = (e2v, e1u)` where `e1u` is the distance in the 1st (x)
        dimension on the U grid and `e2v` is the distance in the 2nd (y)
        dimension on the V grid. These can be of shape `(nj, ni)` or be
        of shape `(nj+1, ni+2)` as in the mesh_mask data file.

    Returns
    -------
    edgedata : array
        1D array of `data`.
        If `a, b = _build_edges(dims)`, then `edgedata[i]` is the
        value from `data` that corresponds to the interface between the grid
        cells indexed by `a[i]` and `b[i]`.
    """

    nj, ni = dims
    v, u = data
    if hasattr(v, 'shape') and v.shape == (nj + 1, ni + 2):
        v = v[:-1, 1:-1]
    if hasattr(u, 'shape') and  u.shape == (nj + 1, ni + 2):
        u = u[:-1, 1:-1]
    v = np.broadcast_to(v, dims)
    u = np.broadcast_to(u, dims)

    return np.concatenate((v.reshape(-1), u.reshape(-1)))


def edgedata_to_maps(edgedata, dims):
    """
    Convert 1D array of data living on edges into two 2D arrays, one for each
    spatial dimension, for a tripolar grid

    Parameters
    ----------
    edgedata : array
        1D array of data that lives on edges of the grid's graph, given in the
        same order as the edges cosntructed from `_build_edges`.
        If `a, b = _build_edges(dims, periodic)`, then `edgedata[i]` is the
        data that lives at the interface between the grid cells indexed by
        `a[i]` and `b[i]`.

    dims :
        See `build_grid`

    Returns
    -------

    v, u : 2D array
        2D arrays of shape `dims`. `v` and `u` contains the values from `edgedata`
        that correspond to data living on the V and U grids, respectively.
    """

    nj, ni = dims
    Ej = ni * nj  # number of edges in j dimension
    v = np.reshape(edgedata[:Ej], dims)
    u = np.reshape(edgedata[Ej:], dims)

    return v, u
