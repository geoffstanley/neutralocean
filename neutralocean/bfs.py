import numpy as np
import numba as nb

from neutralocean.traj import _ntp_bottle_to_cast


@nb.njit
def bfs_conncomp1(G, A, r):
    """
    Find the Connected Component containing 1 reference location using
    Breadth-first Search

    Parameters
    ----------
    G : ndarray of bool

        A 1D array of logicals.  `G[m]` is True iff element `m` is a valid node.
        Note, this function mutates `G`!

    A : ndarray of int

        A 2D array of integers specifying (directional) edges in the graph.
        Consider some valid node `m` (`G[m] == True).  Node `m` is connected
        to node `n = A[m,j]` provided `n < N` and `G[n] == True`, for each
        `j` in `0` to `D-1`.  Here, `N = G.size` is the maximum possible
        number of nodes (if all of `G` is True), and `D = A.shape[-1]` is the
        maximal degree in the graph.  Hence, if node `m` has less than the
        maximal degree (e.g. to handle non-periodic boundaries in regular
        grids, or for more general graphs), some elements of `A[m,:]` should
        be `N`.

    r : int

        Index to the chosen root node, where the BFS begins.

    Returns
    -------
    qu : ndarray of int

        The BFS search queue: a 1D array with as many elements as in `G`.
        `qu[0:qt]` are the indices to the True elements of `G` that are in the
        connected component containing the root node `r`, given in the order
        that the BFS discovered them. For instance, `qu[0] == r`, provided
        the root node is valid (`G[r] == True`). Elements after `qt`, i.e.
        `qu[qt+1 : -1]`, are meaningless.

    qt : int

        The queue tail, i.e. the index to the last meaningful entry in `qu`.
        Hence, `qt+1` is the number of valid nodes in the connected region
        containing the root node. If the root node is invalid (`G[r] ==
        False`) then `qt == -1`.

    """

    N = len(G)
    qu = np.empty(N, dtype=np.int64)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head

    D = A.shape[-1]  # maximal degree

    # Check that the root node is valid.  If not, leave qt as -1, so qu[0:qt+1]
    # is empty
    if G[r]:

        # Initialize BFS from root node
        qt += 1  # Add r to queue
        qu[qt] = r
        G[r] = False  # mark r as discovered

        while qt > qh:
            qh += 1  # advance head of the queue
            m = qu[qh]  # me node; pop from head of queue
            for d in range(D):
                n = A[m, d]  # neighbour node
                if n < N and G[n]:  # First condition checks n is not a
                    #                 neighbour across a non-periodic boundary
                    # n is good and undiscovered
                    qt += 1  # Add n to queue
                    qu[qt] = n
                    G[n] = False  # mark n as discovered

    return qu, qt


@nb.njit
def bfs_conncomp1_wet(s, t, p, S, T, P, n_good, A, r, tol_p, eos, ppc_fn, p_ml):
    """
    As in bfs_conncomp1 but extending the perimeter via wetting

    A breadth-first search begins from the root node `r`, extending through
    wet surface points, i.e. points where `p` is finite.  When an invalid
    node is reached (a dry water column), a neutral tangent plane calculation
    is performed from the surface to this water column.  If successful, and
    if the NTP link reaches this neighbouring cast below the mixed layer,
    then this water column is made "wet": the surface is extended to the
    location on this water column at which NTP intersected, and it is added
    to the BFS so that wetting can proceed as far as possible.

    Parameters
    ----------
    s, t, p : ndarray

        2D practical / Absolute salinity and potential / Conservative
        temperature and pressure / depth on the surface.

        Note, this function mutates `s`, `t`, and `p`!

    S, T, P : ndarray

        3D practical / Absolute salinity and potential / Conservative
        temperature and pressure / depth in the ocean

    n_good : ndarray

        Pre-computed number of ocean data points in each water column.
        This should be computed as ``n_good = lib.find_first_nan(S)``.

    A : ndarray of int
         As in `bfs_conncomp1`

    r : int
        As in `bfs_conncomp1`

    tol_p : float

        Error tolerance when root-finding to update the pressure or depth of
        the surface in each water column. Units are the same as `P`.

    eos : function

        The equation of state, giving the density or specific volume as a
        function of `S`, `T`, and `P` inputs.

        This should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    ppc_fn : function

        Function to compute piecewise polynomial coefficients for an interpolator.
        Construct this as
        `neutralocean.ppinterp.select_ppc("linear", kind="1")`
        for linear interpolation.  For other interpolants, replace "linear"
        (see `select_ppc` documentation).

    p_ml : ndarray

        Pressure or depth of the base of the mixed layer.
        If NTP links that enter the mixed layer are to be retained, then pass
        `p_ml = np.full_like(n_good, -np.inf)`.


    Returns
    -------
    qu : ndarray
        As in `bfs_conncomp1`
    qt : int
        As in `bfs_conncomp1`.
    newly_wet : int
        Number of newly wet water columns.

    """

    ni, nj, nk = S.shape
    N = ni * nj

    qu = np.empty(N, dtype=np.int64)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head

    D = A.shape[-1]  # maximal degree

    G = np.isfinite(p)  # Good nodes

    # Try wetting only these locations: ocean and not currently in the surface
    dry = (n_good > 1) & ~G

    # Flatten lateral dimension of inputs to be 1D.  Use reshape() to get a view.
    G = np.reshape(G, N)
    dry = np.reshape(dry, N)
    s = np.reshape(s, N)
    t = np.reshape(t, N)
    p = np.reshape(p, N)
    S = np.reshape(S, (N, nk))
    T = np.reshape(T, (N, nk))
    P = np.reshape(P, (N, nk))
    n_good = np.reshape(n_good, -1)

    p_ml = np.reshape(p_ml, -1)  # flatten

    # Initialize BFS from root node
    qt += 1  # Add r to queue
    qu[qt] = r
    G[r] = False  # mark r as discovered

    newly_wet = 0
    while qt > qh:
        qh += 1  # advance head of the queue
        m = qu[qh]  # me node; pop from head of queue
        for d in range(D):
            n = A[m, d]  # neighbour node
            if n < N:  # check n is not a neighbour across a non-periodic boundary
                if G[n]:
                    # n is good and undiscovered

                    qt += 1  # Add n to queue
                    qu[qt] = n
                    G[n] = False  # mark n as discovered

                elif dry[n]:
                    # n is "dry".  Try wetting.

                    Sppc = ppc_fn(P[n], S[n])
                    Tppc = ppc_fn(P[n], T[n])

                    s[n], t[n], p[n] = _ntp_bottle_to_cast(
                        s[m],
                        t[m],
                        p[m],
                        Sppc,
                        Tppc,
                        P[n],
                        n_good[n],
                        tol_p,
                        eos,
                    )

                    if np.isfinite(p[n]) and p[n] > p_ml[n]:
                        # The NTP connection was successful, and its location
                        # on the neighbouring cast is below the mixed layer.
                        qt += 1  # Add n to queue
                        qu[qt] = n
                        G[n] = False  # mark n as discovered
                        dry[n] = False
                        newly_wet += 1  # augment counter of newly wet casts

    return qu, qt, newly_wet


def grid_adjacency(dims, conn, wrap):
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

    wrap : tuple of bool
        The periodicity of the grid.  Dimension `i` is periodic iff
        `wrap[i] == True`.


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
    assert ndim == 2, "grid_adjacency currently only works for 2D grids."
    assert len(wrap) == ndim, "wrap must be a tuple the same length as dims."

    ni = dims[0]
    nj = dims[1]

    wallval = ni * nj

    # fmt: off
    # Build adjacency matrix and handle periodicity
    if conn == 4:
        # . 1 .
        # 0 . 3
        # . 2 .
        order = [1, 3, 5, 7]
        adj = _grid_adj_helper(ni, nj, order)

        if not wrap[0]:
            adj[0   , :, 1] = wallval # i-1 hits a wall when i = 0
            adj[ni-1, :, 2] = wallval # i+1 hits a wall when i = ni - 1
        if not wrap[1]:
            adj[:, 0   , 0] = wallval # j-1 hits a wall when j = 0
            adj[:, nj-1, 3] = wallval # j+1 hits a wall when j = nj - 1

    elif conn == 5:
        # . 1 .
        # 0 2 4
        # . 3 .
        # order = [1, 3, 4, 5, 7]

        # . 1 .
        # 0 4 3
        # . 2 .
        order = [1, 3, 5, 7, 4]
        adj = _grid_adj_helper(ni, nj, order)

        if not wrap[0]:
            # adj[0   , :, 1] = wallval # i-1 hits a wall when i = 0
            # adj[ni-1, :, 3] = wallval # i+1 hits a wall when i = ni - 1
            adj[0   , :, 1] = wallval # i-1 hits a wall when i = 0
            adj[ni-1, :, 2] = wallval # i+1 hits a wall when i = ni - 1
        if not wrap[1]:
            # adj[:, 0   , 0] = wallval # j-1 hits a wall when j = 0
            # adj[:, nj-1, 4] = wallval # j+1 hits a wall when j = nj - 1
            adj[:, 0   , 0] = wallval # j-1 hits a wall when j = 0
            adj[:, nj-1, 3] = wallval # j+1 hits a wall when j = nj - 1

    elif conn == 8:
        # 4 1 6
        # 0 . 3
        # 5 2 7
        order = [1, 3, 5, 7, 0, 2, 6, 8]
        adj = _grid_adj_helper(ni, nj, order)

        if not wrap[0]:
            adj[0   , :, [1, 4, 6]] = wallval # i-1 hits a wall when i = 0
            adj[ni-1, :, [2, 5, 7]] = wallval # i+1 hits a wall when i = ni - 1
        if not wrap[1]:
            adj[:, 0   , [0, 4, 5]] = wallval # j-1 hits a wall when j = 0
            adj[:, nj-1, [3, 6, 7]] = wallval # j+1 hits a wall when j = nj - 1

    elif conn == 9:
        # 0 3 6
        # 1 4 7
        # 2 5 8
        order = range(9)
        adj = _grid_adj_helper(ni, nj, order)

        if not wrap[0]:
            adj[0   , :, [0, 3, 6]] = wallval # i-1 hits a wall when i = 0
            adj[ni-1, :, [2, 5, 8]] = wallval # i+1 hits a wall when i = ni - 1
        if not wrap[1]:
            adj[:, 0   , [0, 1, 2]] = wallval # j-1 hits a wall when j = 0
            adj[:, nj-1, [6, 7, 8]] = wallval # j+1 hits a wall when j = nj - 1

    else:
        raise("Unknown number of neighbours.  conn must be one of 4, 5, 8, or 9.")
    # fmt: on

    # Reshape adj to a matrix of dimensions (conn, ni*nj)
    adj = adj.reshape((ni * nj, conn))

    return adj


def _grid_adj_helper(ni, nj, order):

    D = len(order)

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
