import numpy as np
import xarray as xr
import xgcm


def build_grid(
    n, face_connections, dims, xsh, ysh, dxC=1.0, dyC=1.0, dyG=1.0, dxG=1.0
):
    """
    Make a list of edges between adjacent nodes in a graph consisting of
    rectilinear tiles, with adjacency between tiles specified as for xgcm;
    also generate lists of distances associated with these edges.

    Here, the grid is thought of as a graph, where each node represents a
    water column, and the graph has an edge between two nodes when the
    corresponding water columns are adjacent in physical space.


    Parameters
    ----------
    n : int
        Number of grid points on each side of a square tile.
        (Note xgcm only handles square tiles.)

    face_connections : dict
        xgcm-style specification for how tiles (faces) are connected.
        See https://xgcm.readthedocs.io/en/latest/grid_topology.html

    dims : tuple of str
        A tuple of length 3, with one entry being the same as the only key in
        the `face_connections` dict, and the other two being 'i' and 'j'.

        This names, in order, the dimensions of a DataArray whose data lives
        on each (tracer) grid point in the horizontal, and has no vertical
        dimension.

        If `D` is a DataArray giving the Sea Surface Temperature, for example,
        then this argument should be `D.dims`.

        Example: in ECCOv4r4, `dims == ('tile', 'j', 'i')`.

    xsh : str
        The direction of shifting in the 'i' dimension.  Can be either "left"
        or "right".

        If "left", then `dxC` is the distance between the local grid point (i, j)
        and the previous grid point in the 'i' dimension, at (i-1, j).

        If "right", then `dxC` is the distance between the local grid point (i, j)
        and the next grid point in the 'i' dimension, at (i+1, j).

    ysh : str
        The direction of shifting in the 'j' dimension.  Can be either "left"
        or "right".

        If "left", then `dyC` is the distance between the local grid point (i, j)
        and the previous grid point in the 'j' dimension, at (i, j+1).

        If "right", then `dyC` is the distance between the local grid point (i, j)
        and the next grid point in the 'j' dimension, at (i, j+1).

    dxC : float or ndarray, Default 1.0
        Distance between adjacent grid points in the 'i' dimension.

    dyC : float or ndarray, Default 1.0
        Distance between adjacent grid points in the 'j' dimension

    dyG : float or ndarray, Default 1.0
        Distance (in the 'j' dimension) of the face between grid points that
        are adjacent in the 'i' dimension.
        Lives at the same location as `dxC`.

    dxG : float or ndarray, Default 1.0
        Distance (in the 'i' dimension) of the face between grid points that
        are adjacent in the 'j' dimension.
        Lives at the same location as `dyC`.


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

    # Process inputs
    assert xsh[0] in ("l", "r"), "Expected xsh to be either 'left' or 'right'"
    assert ysh[0] in ("l", "r"), "Expected ysh to be either 'left' or 'right'"

    # If given DataArrays, check the dims are correct then convert to numpy arrays.
    if all(hasattr(x, "dims") for x in (dxC, dyC, dyG, dxG)):
        assert (
            dxC.dims == dyG.dims
        ), "Expected dxC and dyG to have the same coordinates"
        assert (
            dyC.dims == dxG.dims
        ), "Expected dyC and dxG to have the same coordinates"
    dxC, dyC, dyG, dxG = (
        x.values if hasattr(x, "values") else x for x in (dxC, dyC, dyG, dxG)
    )

    im1, jm1 = _build_im1_jm1(n, face_connections, dims, xsh, ysh)
    N = im1.size

    grid = dict()

    # Build list of pairs of adjacent nodes.  The first N entries are [m,n]
    # where m is the local node and n is its neighbour in the j dimension.
    # The second N entries are [m,n] where m is the local node and n is its
    # neighbour in the i dimension.
    a = np.tile(np.arange(N), 2)
    b = np.empty(N * 2, dtype=int)  # prealloc space
    b[:N] = im1.values.reshape(-1)
    b[N:] = jm1.values.reshape(-1)

    # Build list of distances between adjacent nodes, in the same order as the
    # pairs of nodes built by edges.
    dist = np.empty(N * 2, dtype=float)
    dist[:N] = dyC.reshape(-1)
    dist[N:] = dxC.reshape(-1)

    # Build list of distances of the faces between adjacent nodes, in the same
    # order as the pairs of nodes built by edges.
    distperp = np.empty(N * 2, dtype=float)
    distperp[:N] = dxG.reshape(-1)  # dxG lives at same place as dyC
    distperp[N:] = dyG.reshape(-1)  # dyG lives at same place as dxC

    # Trim out invalid edges, i.e. those for which one of the two nodes was
    # filled by `apply_as_grid_ufunc` to have a value of -1.
    good = b >= 0
    a, b, dist, distperp = (x[good] for x in (a, b, dist, distperp))

    grid["edges"] = (a, b)
    grid["dist"] = dist
    grid["distperp"] = distperp
    return grid


def edgedata_to_maps(edgedata, n, face_connections, dims, xsh, ysh):
    """
    Convert 1D array of data living on edges into two nD arrays, one for each
    spatial dimension, for a rectilinear grid

    Parameters
    ----------
    edgedata : array
        1D array of data that lives on edges of the grid's graph, given in the
        same order as the edges cosntructed from `build_grid`.
        If `a, b, ... = build_grid(...)`, then `edgedata[i]` is the
        data that lives at the interface between the grid cells indexed by
        `a[i]` and `b[i]`.

    n, face_connections, dims, xsh, ysh :
        See `build_grid`

    Returns
    -------
    Fi, Fj : ndarray
        `Fi` contains the values from `edgedata` that correspond to data living
        between grid points in the first dimension, and similarly for `Fj` but
        in the second dimension.
    """

    im1, jm1 = _build_im1_jm1(n, face_connections, dims, xsh, ysh)
    N = im1.size

    Fi, Fj = (np.full(N, np.nan) for x in (0, 1))

    # Build linear indices for grid cells (i,j) where (i-1,j) is valid
    idx_goodim1 = np.flatnonzero(im1 >= 0)
    Ni = len(idx_goodim1)

    Fi[idx_goodim1] = edgedata[:Ni]
    Fj[np.flatnonzero(jm1 >= 0)] = edgedata[Ni:]

    Fi, Fj = (np.reshape(x, im1.shape) for x in (Fi, Fj))
    return Fi, Fj


def _build_im1_jm1(n, face_connections, dims, xsh, ysh):

    # Access first (and only) key / value of face_connections
    tile = next(
        iter(face_connections.keys())
    )  # name of face_connections only key
    nf = len(next(iter(face_connections.values())))  #  number of faces

    # Get index of tile in dims
    try:
        t = dims.index(tile)
    except:
        raise ValueError(
            f"Expected to find '{tile}' (the key from `face_connections`) in `dims` == {dims}."
        )

    # Ensure dims also has 'i' and 'j':
    if not (dims.__contains__("i") and dims.__contains__("j")):
        raise ValueError(f"Expected to find 'i' and 'j' in `dims` == {dims}.")

    N = nf * n * n  # number of Nodes (water columns)

    # Shape of water columns (e.g. shape of ndarray storing sea surface temperature data)
    shape = [n, n, n]
    shape[t] = nf

    # Build a simple xgcm grid
    coords = {"i": np.arange(n), "j": np.arange(n), tile: np.arange(nf)}
    grid = xgcm.Grid(
        xr.Dataset(None, coords),
        periodic=False,
        face_connections=face_connections,
        coords={
            "X": {"center": "i", "left": "i", "right": "i"},
            "Y": {"center": "j", "left": "j", "right": "j"},
        },
    )

    # Make linear indices for the whole domain: a DataArray of 0, 1, ... N-1.
    idx = xr.DataArray(np.arange(N).reshape(shape), dims=dims, coords=coords)

    # Make linear indices to the neighbour in the i//x dimension.
    # That is, idx[t,j,i-1] == im1[t,j,i] for all 0 <= t < nf, 0 <= j < n, 0 < i < n
    # and corresponding treatment across tiles when i == 0.
    # Similarly, idx[t,j-1,i] == jm1[t,j,i] for all 0 <= t < nf, 0 < j < n, 0 <= i < n
    # and corresponding treatment across tiles when j == 0.
    # To achieve this, use XGCM grid Ufunc to pad the `idx` data as needed,
    # then subset forwards or backwards.
    if xsh[0] == "l":  # left
        # # Old code using depreciated XGCM function below
        # im1 = grid.axes["X"]._neighbor_binary_func(
        #     idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
        # )
        im1 = grid.apply_as_grid_ufunc(
            lambda a: a[:, :, 0:-1],
            idx,
            axis=[["X"]],
            signature="(ax1:center)->(ax1:center)",
            boundary_width={"ax1": (1, 0)},
            boundary="fill",
            fill_value=-1,
        )

    else:  # right
        # im1 = grid.axes["X"]._neighbor_binary_func(
        #     idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
        # )
        im1 = grid.apply_as_grid_ufunc(
            lambda a: a[:, :, 1:],
            idx,
            axis=[["X"]],
            signature="(ax1:center)->(ax1:center)",
            boundary_width={"ax1": (0, 1)},
            boundary="fill",
            fill_value=-1,
        )

    # Make linear indices to the neighbour in the j//y dimension.
    # Why do the following lambda functions subset along the last dimension,
    # when j is the second (middle) dimension?  See:
    # https://xgcm.readthedocs.io/en/latest/grid_ufuncs.html
    # "XGCM assumes the function acts along the last axis of the numpy array"
    if ysh[0] == "l":  # left
        # jm1 = grid.axes["Y"]._neighbor_binary_func(
        #     idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
        # )
        jm1 = grid.apply_as_grid_ufunc(
            lambda a: a[:, :, 0:-1],
            idx,
            axis=[["Y"]],
            signature="(ax1:center)->(ax1:center)",
            boundary_width={"ax1": (1, 0)},
            boundary="fill",
            fill_value=-1,
        )

        # Must ensure same ordering of dimensions as idx.  Currently, this
        # one gets its dimensions reordered by the grid ufunc, probably
        # because it reorders things to act on the last dimension...?
        jm1 = jm1.transpose(*idx.dims)

    else:  # right
        # jm1 = grid.axes["Y"]._neighbor_binary_func(
        #     idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
        # )
        jm1 = grid.apply_as_grid_ufunc(
            lambda a: a[:, :, 1:],
            idx,
            axis=[["Y"]],
            signature="(ax1:center)->(ax1:center)",
            boundary_width={"ax1": (0, 1)},
            boundary="fill",
            fill_value=-1,
        )

    return im1, jm1
