import numpy as np
import xarray as xr
import xgcm


def build_edges_and_geometry(n, face_connections, dims, dxC, dyC, dxG, dyG, xsh, ysh):
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
        This names, in order, the dimensions of a DataArray whose data lives
        on each (tracer) grid point in the horizontal, and has no vertical
        dimension.

        If `D` is a DataArray giving the Sea Surface Temperature, for example,
        then this argument should be `D.dims`.

        This must be a tuple of length 3.  One of the entries must
        be the same as the only key in the `face_connections` dict.  The other
        two entires must be 'i' and 'j'.  We need `dims` as an input here
        because the order matters.

        An example: in ECCOv4r4, `dims == ('tile', 'j', 'i')`.

    dxC : ndarray
        Distance between adjacent grid points in the 'i' dimension.

    dyC : ndarray
        Distance between adjacent grid points in the 'j' dimension

    dxG : ndarray
        Distance of the face between adjacent grid points in the 'j' dimension

    dyG : ndarray
        Distance of the face between adjacent grid points in the 'i' dimension

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


    Returns
    -------
    edges : 2d array of int
        Array of shape (E, 2), where E is the number of edges in the grid's graph,
        i.e. the number of pairs of adjacent water columns in the grid.
        For each i, edges[i,:] is a 2-element array giving linear indices to
        a pair of adjacent nodes (water columns).

    dist : 1d array
        Distance between adjacent grid points.  `dist[m]` is the distance
        between nodes (water columns) whose linear indices are `edges[m,0]` and
        `edges[m,1]`.

    distperp : 1d array
        Distance of the face between adjacent grid points.
        `distperp[m]` is the distance of the face between nodes (water columns)
        whose linear indices are `edges[m,0]` and `edges[m,1]`.

    """

    assert xsh[0] in ("l", "r"), "Expected xsh to be either 'left' or 'right'"
    assert ysh[0] in ("l", "r"), "Expected ysh to be either 'left' or 'right'"

    # If given DataArrays, check the dims are correct then convert to numpy arrays.
    if all(hasattr(x, "dims") for x in (dxC, dyC, dxG, dyG)):
        assert dxC.dims == dyG.dims, "Expected dxC and dyG to have the same coordinates"
        assert dyC.dims == dxG.dims, "Expected dyC and dxG to have the same coordinates"
    dxC, dyC, dxG, dyG = (
        x.values if hasattr(x, "values") else x for x in (dxC, dyC, dxG, dyG)
    )

    # Access first (and only) key / value of face_connections
    tile = next(iter(face_connections.keys()))  # name of face_connections only key
    nf = len(next(iter(face_connections.values())))  #  number of faces

    # Get index of tile in dims
    try:
        t = dims.index(tile)
    except:
        raise ValueError(
            f"Expected to find '{tile}' (the key from `face_connections`) in `dims` == {dims}."
        )

    N = n * n * nf  # number of Nodes (water columns)

    # Shape of water columns (e.g. shape of ndarray storing sea surface temperature data)
    shape = [n, n, n]
    shape[t] = nf

    # Build a simple xgcm grid
    coords = {"i": np.arange(n), "j": np.arange(n), tile: np.arange(nf)}
    gg = xr.Dataset(None, coords)
    grid = xgcm.Grid(
        gg,
        periodic=False,
        face_connections=face_connections,
        coords={
            "X": {"center": "i", "left": "i", "right": "i"},
            "Y": {"center": "j", "left": "j", "right": "j"},
        },
    )

    # Make linear indices for the whole domain: a DataArray of 0, 1, ... N-1.
    idx = xr.DataArray(np.arange(N).reshape(shape), dims=dims, coords=coords)

    # Make linear indices to the neighbour in the i//x dimension
    if xsh[0] == "l":  # left
        ish = grid.axes["X"]._neighbor_binary_func(
            idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
        )
    else:  # right
        ish = grid.axes["X"]._neighbor_binary_func(
            idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
        )

    # Make linear indices to the neighbour in the j//y dimension
    if ysh[0] == "l":  # left
        jsh = grid.axes["Y"]._neighbor_binary_func(
            idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
        )
    else:  # right
        jsh = grid.axes["Y"]._neighbor_binary_func(
            idx, lambda a, b: b, to="right", boundary="fill", fill_value=-1
        )

    # Build list of pairs of adjacent nodes.  The first N entries are [m,n]
    # where m is the local node and n is its neighbour in the j dimension.
    # The second N entries are [m,n] where m is the local node and n is its
    # neighbour in the i dimension.
    edges = np.empty((N * 2, 2), dtype=int)
    edges[:, 0] = np.tile(np.arange(N), 2)
    edges[:N, 1] = jsh.values.reshape(-1)
    edges[N:, 1] = ish.values.reshape(-1)

    # Build list of distances between adjacent nodes, in the same order as the
    # pairs of nodes built by edges.
    dist = np.empty(N * 2, dtype=float)
    dist[:N] = dyC.reshape(-1)
    dist[N:] = dxC.reshape(-1)

    # Build list of distances of the faces between adjacent nodes, in the same
    # order as the pairs of nodes built by edges.
    distperp = np.empty(N * 2, dtype=float)
    distperp[:N] = dxG.reshape(-1)
    distperp[N:] = dyG.reshape(-1)

    # Trim out invalid edges, i.e. those for which one of the two nodes was
    # filled by `_neighbor_binary_func`.
    good = np.all(edges >= 0, axis=1)
    edges = edges[good, :]
    dist = dist[good]
    distperp = distperp[good]

    return edges, dist, distperp
