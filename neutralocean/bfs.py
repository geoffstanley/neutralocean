import numpy as np
import numba as nb

from .traj import _ntp_bottle_to_cast, _ntp_bottle_to_cast_ppc
from .ppinterp import valid_range_1_two, ppval_1_two
from .lib import local_functions


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
    qu = np.empty(N, dtype=np.int_)

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
def bfs_conncomp1_wet(s, t, p, indptr, indices, root, S, T, P, tol_p, eos, ppc_fn, p_ml):
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

    Notes
    -----
    This function only performs at most one NTP bottle to cast calculation per
    dry cast. When this cast is adjacent to multiple wet casts, this means
    there will be one perfectly neutral link, and other links will be
    non-neutral. As such, the output and the mutated values of `s`, `t`, `p`
    are sensitive to the order in which the neighbours of each node (cast) are
    listed in `indptr` and `indices`. Therefore, it is adviseable to construct
    `indptr` and `indices` using `edges_to_csr` rather than `edges_to_graph`
    from `neutralocean.grid.graph`. See Notes of `edges_to_csr`.

    """

    N = S.shape[0]  # Number of nodes (water columns)

    qu = np.empty(N, dtype=np.int_)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head
    newly_wet = 0

    good = np.isfinite(p)  # Good nodes
    dry = ~good  # nodes to try wetting

    if good[root]:
        # Initialize BFS from root node
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

                    # Try NTP link from bottle at m to cast at n.
                    # TODO: impose p[n] <= p_ml[n] here instead of below?
                    s_, t_, p_ = _ntp_bottle_to_cast(
                        s[m], t[m], p[m], Sn, Tn, Pn, k, K, tol_p, eos, ppc_fn
                    )

                    if np.isfinite(p_) and p_ > p_ml[n]:
                        # The NTP connection was successful, and its location
                        # on the neighbouring cast is below the mixed layer.
                        qt += 1  # Add n to queue
                        qu[qt] = n
                        dry[n] = False  # mark n as wet
                        newly_wet += 1  # augment counter of newly wet casts
                        s[n], t[n], p[n] = s_, t_, p_

    return qu[0 : qt + 1], newly_wet


@nb.njit
def bfs_conncomp1_wet_perim(
    s, t, p, indptr, indices, root, S, T, P, tol_p, eos, ppc_fn, p_ml, n_rep
):
    """
    Find the Connected Component of a graph's good nodes containing a root node
    via Breadth-first Search.
    Repeatedly wet the perimeter of this connected component.

    Parameters
    ----------
    s, t, p, indptr, indices, root, S, T, P, tol_p, eos, ppc_fn, p_ml :
        See `bfs_conncomp1_wet`

    n_rep : int
        Maximum number of times to repeat wetting at the perimeter.
        To perform wetting as many times as necessary until no further NTP
        links are possible, give n_rep as a very large int, e.g.
        `n_rep = numpy.iinfo(np.int_).max`


    Returns
    -------
    newly_wet : int
        Number of newly wet water columns.

    Notes
    -----
    To get indices to the wet casts, just use
        idx = np.nonzero(np.isfinite(p))[0]
    The queue that this method uses internally contains both wet casts and
    dry casts on the perimeter of the wet region, so this queue cannot be used
    to get a list of only wet casts. The slight cost of calling np.nonzero
    may be counterbalanced by the fact that it gives a sorted list of
    indices, whereas the `qu` from `bfs_conncomp1_wet` is not sorted.

    Because this function performs NTP bottle to cast calculations from a dry
    cast to all adjacent wet casts, the output and mutated values of `s`, `t`,
    `p` are NOT sensitive (to machine precision) to the order in which the
    neighbours of each node (cast) are listed in `indptr` and `indices`.
    Therefore, one can construct `indptr` and `indices` using EITHER
    `edges_to_csr` or `edges_to_graph` from `neutralocean.grid.graph`.

    """

    good = np.isfinite(p)  # Good nodes
    wet_adj = good.copy()  # True at n if n is wet or neighbour of a wet cast
    N = len(good)  # number of nodes in entire grid
    qu = np.empty(N, dtype=np.int_)
    qt = -1  # Queue Tail
    qh = -1  # Queue Head
    qH = N - 1  # Queue head for perimeter nodes
    qT = N - 1  # Queue Tail for perimeter nodes
    newly_wet = 0

    if good[root]:
        # Root node is valid.  Initialize BFS from root node
        qt += 1  # Add root to queue
        qu[qt] = root
        good[root] = False  # mark root as discovered

    # else, root node is invalid.  Leave qt as -1, so qu[0:qt+1] is empty

    while qt > qh:
        qh += 1  # advance head of the queue
        m = qu[qh]  # pop node m from head of queue

        # loop over neighbour nodes
        for n in indices[indptr[m] : indptr[m + 1]]:
            if good[n]:
                # n is good and undiscovered
                qt += 1  # Add n to queue for BFS from root
                qu[qt] = n
                good[n] = False  # mark n as discovered
            elif not wet_adj[n]:
                # n is not good and undiscovered
                qu[qT] = n  # Add n to list of perimeter nodes
                qT -= 1
                wet_adj[n] = True

    # Set p to nan everywhere beyond the connected component containing root.
    good[:] = False
    for i in range(qt + 1):
        good[qu[i]] = True
    p[~good] = np.nan

    # Henceforth, we can use good to record ocean vs land. good[m] is True when
    # m is an ocean cast with two or more grid points, False otherwise.
    # However, there's no discernable speedup of doing so.
    # good[:] = True  # assume everything ocean to start

    rep = 0
    while qH > qT and rep < n_rep:
        rep += 1

        # Loop over perimeter casts
        for i in range(qH, qT, -1):
            m = qu[i]  # dry cast
            # if good[m]:  # ocean with at least 2 grid points

            # assume S and T have same nan-structure
            k, K = valid_range_1_two(S[m], P[m])

            if K - k > 1:
                # Cast m is ocean with 2+ grid points. Build interpolants
                Pm = P[m][k:K]
                Sppc = ppc_fn(Pm, S[m][k:K])
                Tppc = ppc_fn(Pm, T[m][k:K])

                # Loop over neighbour nodes, looking for a wet cast to NTP to
                neigh = indices[indptr[m] : indptr[m + 1]]
                p0 = 0.0  # for accumulating pressures
                denom = 0  # number of successful NTP links to this cast
                for n in neigh:
                    if np.isfinite(p[n]):
                        # Try NTP link from bottle at n to cast at m
                        p_ = _ntp_bottle_to_cast_ppc(
                            tol_p, s[n], t[n], p[n], Pm, Sppc, Tppc, eos
                        )

                        if np.isfinite(p_) and p_ > p_ml[n]:
                            # The NTP connection was successful, and its location
                            # on the neighbouring cast is below the mixed layer.
                            p0 += p_
                            denom += 1

                if denom != 0:
                    newly_wet += 1  # augment counter of newly wet casts

                    # Average pressures of all successful NTP links
                    p[m] = p0 / denom

                    # Interpolate S and T to averaged pressure
                    s[m], t[m] = ppval_1_two(p[m], Pm, Sppc, Tppc, 0)

                    # Cast m has been wet, so add its neighbours that are
                    # not (wet or adjacent to the wet region) to the queue.
                    for n in neigh:
                        if not wet_adj[n]:
                            qu[qT] = n  # Add n to list of nodes on perimeter
                            qT -= 1
                            wet_adj[n] = True

            # else:
            #     good[m] = False  # don't try here again

        qH = i - 1  # Reset for next pass over perimeter

    return newly_wet


__all__ = local_functions(locals(), __name__)
