import numpy as np
import numba as nb

from neutralocean.traj import _ntp_bottle_to_cast
from neutralocean.ppinterp import valid_range_1_two


@nb.njit
def bfs_conncomp1(indptr, indices, root, good):
    """
    Find the Connected Component containing 1 reference location using
    Breadth-first Search

    Parameters
    ----------
    indptr, indices : ndarray of int

        Together, these specify the connectivity of the full graph.  Node `m` is
        adjacent to the nodes `indices[indptr[m] : indptr[m+1]]`.  Note that
        these are simply the `indptr` and `indices` attributes of a the graph
        specified as a sparse matrix in csr_matrix format, i.e. a square matrix
        `G` that has `G[i,j] != 0` whenever nodes `i` and `j` are adjacent.

    root : int

        Index to the chosen root node, where the BFS begins.

    good : ndarray of bool

        A 1D array of logicals for masking out certain nodes in the graph.
        The BFS only operates on "valid nodes" `m` for which `good[m]` is True.
        Note, this function mutates `good`!

    Returns
    -------
    qu : ndarray of int

        The BFS search queue.  A 1D array giving the indices to nodes in the
        connected component containing the `root` node, in the order that the
        BFS discovered them.
        If the root node is valid (`good[root] == True`), then `qu[0] == root`.
        Otherwise, the BFS goes nowhere and `qu` is an empty array.
    """

    N = len(good)
    qu = np.empty(N, dtype=np.int64)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head

    if good[root]:
        # Root node is valid.  Initialize BFS from root node
        qt += 1  # Add root to queue
        qu[qt] = root
        good[root] = False  # mark root as discovered

        while qt > qh:
            qh += 1  # advance head of the queue
            m = qu[qh]  # me node; pop from head of queue

            # loop over neighbour nodes
            for n in indices[indptr[m] : indptr[m + 1]]:
                if good[n]:
                    # n is good and undiscovered
                    qt += 1  # Add n to queue
                    qu[qt] = n
                    good[n] = False  # mark n as discovered
    # else, root node is invalid.  Leave qt as -1, so qu[0:qt+1] is empty

    return qu[0 : qt + 1]


@nb.njit
def bfs_conncomp1_wet(
    indptr, indices, root, s, t, p, S, T, P, tol_p, eos, ppc_fn, p_ml
):
    """
    As in bfs_conncomp1 but extending the perimeter via wetting.

    A breadth-first search begins from the root node `root`, extending through
    wet surface points, i.e. points where `p` is finite.  When an invalid
    node is reached (a dry water column), a neutral tangent plane calculation
    is performed from the surface to this water column.  If successful, and
    if the NTP link reaches this neighbouring cast below the mixed layer,
    then this water column is made "wet": the surface is extended to the
    location on this water column at which NTP intersected, and it is added
    to the BFS so that wetting can proceed as far as possible.

    Parameters
    ----------
    indptr, indices, root :
         As in `bfs_conncomp1`

    s, t, p : ndarray

        2D practical / Absolute salinity and potential / Conservative
        temperature and pressure / depth on the surface.

        Note, this function mutates `s`, `t`, and `p`!

    S, T, P : ndarray

        3D practical / Absolute salinity and potential / Conservative
        temperature and pressure / depth in the ocean

    tol_p : float

        Error tolerance when root-finding to update the pressure or depth of
        the surface in each water column. Units are the same as `P`.

    eos : function

        The equation of state, giving the density or specific volume as a
        function of `S`, `T`, and `P` inputs.

        This should be `@numba.njit` decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    ppc_fn : function

        Function to compute piecewise polynomial coefficients for an interpolator.
        Construct this as
        `neutralocean.ppinterp.make_pp("linear", kind="1", nans=False)`
        for linear interpolation.  For other interpolants, replace `"linear"`
        (see `make_pp` documentation).

    p_ml : ndarray

        Pressure or depth of the base of the mixed layer.
        If NTP links that enter the mixed layer are to be retained, then pass
        `p_ml = np.full_like(s.shape, -np.inf)`.


    Returns
    -------
    qu : ndarray
        As in `bfs_conncomp1`

    newly_wet : int
        Number of newly wet water columns.

    """

    # N, nk = S.shape
    N = S.shape[0]  # Number of nodes (water columns)

    qu = np.empty(N, dtype=np.int64)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head
    newly_wet = 0

    good = np.isfinite(p)  # Good nodes
    dry = ~good  # nodes to try wetting

    # Flatten lateral dimension of inputs to be 1D.  Use reshape() to get a view.
    # ... removed this!  Should already have been done.
    # good = np.reshape(good, N)
    # dry = np.reshape(dry, N)
    # s = np.reshape(s, N)
    # t = np.reshape(t, N)
    # p = np.reshape(p, N)
    # S = np.reshape(S, (N, nk))
    # T = np.reshape(T, (N, nk))
    # P = np.reshape(P, (N, nk))
    # p_ml = np.reshape(p_ml, -1)  # flatten

    if good[root]:
        # Initialize BFS from root node
        qt += 1  # Add r to queue
        qu[qt] = root
        good[root] = False  # mark r as discovered

        while qt > qh:
            qh += 1  # advance head of the queue
            m = qu[qh]  # me node; pop from head of queue

            # loop over neighbour nodes
            for n in indices[indptr[m] : indptr[m + 1]]:
                if good[n]:
                    # n is good and undiscovered

                    qt += 1  # Add n to queue
                    qu[qt] = n
                    good[n] = False  # mark n as discovered

                elif dry[n]:
                    # n is "dry".  Try wetting.

                    Sn = S[n]
                    Tn = T[n]
                    Pn = P[n]

                    # assume S and T have same nan-structure
                    k, K = valid_range_1_two(Sn, Pn)

                    if K - k <= 1:
                        # At most 1 valid ocean data site.  Can't interpolate
                        # with that.  Mark as not dry, so don't revisit.
                        dry[n] = False
                        continue

                    # Do NTP link from bottle at m to cast at n.
                    s[n], t[n], p[n] = _ntp_bottle_to_cast(
                        s[m], t[m], p[m], Sn, Tn, Pn, k, K, tol_p, eos, ppc_fn
                    )

                    if np.isfinite(p[n]) and p[n] > p_ml[n]:
                        # The NTP connection was successful, and its location
                        # on the neighbouring cast is below the mixed layer.
                        qt += 1  # Add n to queue
                        qu[qt] = n
                        good[n] = False  # mark n as discovered
                        dry[n] = False
                        newly_wet += 1  # augment counter of newly wet casts

    return qu[0 : qt + 1], newly_wet
