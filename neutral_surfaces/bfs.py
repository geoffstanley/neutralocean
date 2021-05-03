import numpy as np
import numba
import functools

from neutral_surfaces.lib import ntp_bottle_to_cast


@numba.njit
def bfs_conncomp1(G, A, r):
    # Note!  Mutates G.   Needs G to be a 1D array.

    # BFS_CONNCOMP1  Find one connected component using Breadth First Search
    #
    #
    # qu, qt = bfs_conncomp(G, A, r)
    # performs a breadth first search (BFS) through nodes of a graph, starting
    # from the root node r.  The graph is best thought of as laid out on a
    # grid, with nodes at the true locations of the multi-dimensional array G
    # and numbered by their linear index in G.   Node m is adjacent to node n =
    # A[m,j] provided (n < G.size and G[n]), for each j in 0 to D-1, where D =
    # A.shape[-1] is the maximal degree in the graph. The outputs are: (1) the search
    # queue for the BFS, qu; (2) the tail index of qu for the BFS, qt.
    # Specifically, qu[0 : qt+1] are the linear indices to True locations of G
    # searched by the BFS.  Note that G is mutated by this function.
    #
    #
    # --- Input:
    # G [array with N elements]: True where there are nodes, False elsewhere
    # A [N,D]: adjacency, where D is the most neighbours possible
    # r [1,1]: perform one BFS from this root node
    #
    #
    # --- Output:
    # qu [N]: the nodes visited by the BFS's in order from 0 to qt
    # qt [1]: the queue tail index
    #
    #
    # See also GRID_ADJACENCY

    # Author[s] : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

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


@numba.njit
def bfs_conncomp1_wet(s, t, p, S, T, P, Sppc, Tppc, n_good, A, r, tol_p, eos):
    # Note! Mutates s, t, p

    ni, nj, nk = S.shape
    nij = ni * nj

    freshly_wet = 0

    qu = np.empty(nij, dtype=np.int64)

    qt = -1  # Queue Tail
    qh = -1  # Queue Head

    D = A.shape[-1]  # maximal degree

    G = np.isfinite(p)  # Good nodes

    # Try wetting only these locations: ocean and not currently in the surface
    dry = (n_good > 1) & ~G

    # Flatten lateral dimension of inputs to be 1D.  Use reshape() to get a view.
    G = np.reshape(G, nij)
    dry = np.reshape(dry, nij)
    s = np.reshape(s, nij)
    t = np.reshape(t, nij)
    p = np.reshape(p, nij)
    S = np.reshape(S, (nij, nk))
    T = np.reshape(T, (nij, nk))
    P = np.reshape(P, (nij, nk))
    Sppc = np.reshape(Sppc, (nij, nk - 1, -1))
    Tppc = np.reshape(Tppc, (nij, nk - 1, -1))
    n_good = np.reshape(n_good, -1)

    # Initialize BFS from root node
    qt += 1  # Add r to queue
    qu[qt] = r
    G[r] = False  # mark r as discovered

    while qt > qh:
        qh += 1  # advance head of the queue
        m = qu[qh]  # me node; pop from head of queue
        for d in range(D):
            n = A[m, d]  # neighbour node
            if n < nij:  # check n is not a neighbour across a non-periodic boundary
                if G[n]:
                    # n is good and undiscovered

                    qt += 1  # Add n to queue
                    qu[qt] = n
                    G[n] = False  # mark n as discovered

                elif dry[n]:
                    # n is "dry".  Try wetting.

                    s[n], t[n], p[n] = ntp_bottle_to_cast(
                        s[m],
                        t[m],
                        p[m],
                        S[n],
                        T[n],
                        P[n],
                        Sppc[n],
                        Tppc[n],
                        n_good[n],
                        tol_p,
                        eos
                    )

                    if np.isfinite(p[n]):
                        # The NTP connection was successful

                        # s[np.unravel_index(n, (ni,nj))] = s1[n]
                        # t[np.unravel_index(n, (ni,nj))] = t1[n]
                        # p[np.unravel_index(n, (ni,nj))] = p1[n]

                        qt += 1  # Add n to queue
                        qu[qt] = n
                        G[n] = False  # mark n as discovered
                        dry[n] = False

                        freshly_wet += 1  # augment counter of freshly wet casts

    # Unflatten to 2D arrays
    # s[:] = np.reshape(ss, (ni,nj))
    # t[:] = np.reshape(tt, (ni,nj))
    # p[:] = np.reshape(pp, (ni,nj))

    return qu, qt, freshly_wet


# function grid_adjacency(dims::Tuple{Int,Int}, conn::Int, wrap::Tuple{Bool,Bool})
def grid_adjacency(dims, conn, wrap):

    # GRID_ADJACENCY  Linear indices to each neighbour of each grid point on a grid
    #
    #
    # adj = grid_adjacency(dims, conn, wrap)
    # builds a matrix adj giving linear indices to each of conn neighbours of
    # each grid point in a grid of size specified by dims.  The grid is periodic
    # in the i'th dimension if and only if wrap[i] is true. The n'th column of
    # adj is a vector of linear indices to grid points that are adjacent to the
    # grid point whose linear index is n.  The maximum number of neighbours of
    # any grid point is conn, which is the number of rows of adj.  If grid
    # point n has fewer neighbours than conn; due to being near a non-periodic
    # boundary, adj[n,:] will contain some flag values.  The flag value is
    # ni*nj, which can be used to index some special value.
    #
    # The connectivity conn specifies the number of neighbours to a central grid
    # point, possibly including itself.  For i between 1 and conn; the i'th
    # neighbour is located relative to the central grid point according to the
    # following diagram:
    #  conn==4        conn==5   conn==8   conn==9
    # +---------> j
    # |   . 1 .       . 1 .     4 1 6     0 3 6
    # |   0 . 3       0 2 4     0 . 3     1 4 7
    # v   . 2 .       . 3 .     5 2 7     2 5 8
    # i
    # Here, i increases downward and j increases right.  For example, if conn
    # == 4, the 2'nd neighbour (indexed by 1) of the central grid point at (i,j) is located at
    # (i-1,j).
    #
    #
    # --- Input:
    # dims: tuple specifying dimensions of the grid
    # conn: the connectivity: 4, 5, 8, or 9.
    # wrap: tuple specifying periodicity of the grid.
    #
    #
    # --- Output:
    # adj [dims, conn]: the adjacency matrix

    # Author[s] : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

    ndim = len(dims)  # Number of dimensions in the grid
    assert ndim == 2, "grid_adjacency currently only works for 2D grids."
    assert len(wrap) == ndim, "wrap must be a tuple the same length as dims."

    ni = dims[0]
    nj = dims[1]

    wallval = ni * nj

    def helper(ni, nj, order):

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

    # fmt: off
    # Build adjacency matrix and handle periodicity
    if conn == 4:
        # . 1 .
        # 0 . 3
        # . 2 .
        order = [1, 3, 5, 7]
        adj = helper(ni, nj, order)

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
        adj = helper(ni, nj, order)

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
        adj = helper(ni, nj, order)

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
        adj = helper(ni, nj, order)

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
