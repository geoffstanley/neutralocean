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
