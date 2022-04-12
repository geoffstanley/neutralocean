import numpy as np
import xarray as xr
import xgcm


def xgcm_to_edges(n, face_connections):
    """
    Make a list of edges between adjacent nodes in a grid's graph as specified by xgcm

    Parameters
    ----------
    n : int
        Number of grid points on each side of a square tile.
        (Note xgcm only handles square tiles.)

    face_connections : dict
        xgcm-style specification for how tiles (faces) are connected.
        See https://xgcm.readthedocs.io/en/latest/grid_topology.html

    Returns
    -------
    edges : 2d array of int
        Array of shape (E, 2), where E is the number of edges in the grid's graph,
        i.e. the number of pairs of adjacent water columns in the grid.
        For each i, edges[i,:] is a 2-element array giving linear indices to
        a pair of adjacent nodes (water columns).
    """

    # Access first (and only) key / value of face_connections
    fname = next(iter(face_connections.keys()))  # name of face_connections only key
    nf = len(next(iter(face_connections.values())))  # len(faces_connections only value)

    gg = xr.Dataset(None, {"i": np.arange(n), "j": np.arange(n), fname: np.arange(nf)})
    grid = xgcm.Grid(
        gg,
        periodic=False,
        face_connections=face_connections,
        coords={
            "X": {"center": "i", "left": "i", "right": "i"},
            "Y": {"center": "j", "left": "j", "right": "j"},
        },
    )

    N = n * n * nf  # number of nodes

    # Make a DataArray but filled with 0, 1, ... N-1
    idx = xr.DataArray(
        np.arange(N).reshape((nf, n, n)),
        dims=(fname, "j", "i"),
        coords=(np.arange(nf), np.arange(n), np.arange(n)),
    )

    # Note j is the Y axis and i is the X axis, when we write data[f, j, i]  (f the face index)
    idx_jm1 = grid.axes["Y"]._neighbor_binary_func(
        idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
    )  # 940 µs
    idx_im1 = grid.axes["X"]._neighbor_binary_func(
        idx, lambda a, b: a, to="left", boundary="fill", fill_value=-1
    )

    edges = np.empty((N * 2, 2), dtype=int)
    edges[:, 0] = np.tile(np.arange(N), 2)
    edges[:N, 1] = idx_jm1.values.reshape(-1)
    edges[N:, 1] = idx_im1.values.reshape(-1)

    good = np.all(edges >= 0, axis=1)
    edges = edges[good, :]

    return edges
