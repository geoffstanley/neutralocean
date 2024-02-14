import numpy as np
import numba as nb

from scipy.sparse import coo_matrix, csr_matrix, find, triu, issparse
from scipy.linalg import issymmetric


@nb.njit
def graph_binary_fcn(edges, nodevals, binary_fcn):
    """
    Apply a binary function to all pairs of data for all pairs of adjacent nodes
    in a graph specified by a sparse COO matrix.

    Parameters
    ----------
    edges : 2D array of shape `(2, E)`, or length 2 tuple of 1D arrays of length `E`

        The edges in a graph whose nodes are labelled `0, ..., N-1`.
        Denoting `a = edges[0]` and `b = edges[1]`,
        - `a` and `b` are 1D arrays of type int
        - the nodes labelled `a[i]` and `b[i]` are adjacent, for `0 <= i <= E-1`
        - must have `0 <= a[i], b[i] <= N-1` for `0 <= i <= E-1`, so `edges` is
        referring to valid nodes.

    nodevals : array-like
        values at the nodes in a graph

    binary_fcn : function
        A `numba.njit` function accepting two numbers and returning one number.

    Notes
    -----
    If `edges` is a 2D array of shape `(E, 2)` or is a tuple of length `E`
    with each element a tuple of int of length 2, then one must convert it
    as follows:

    >>> edges = numpy.array(edges).transpose()

    Returns
    -------
    ev : 1d array
        The value of `binary_fcn` evaluated on each edge.
        That is, `ev[e] = binary_fcn(data[i], data[j])` where `i = row[e]` and
        `j = col[e]`.  If either `i` or `j` is < 0, `ev[e]` is nan.
    """

    a, b = edges

    ev = np.empty(len(a), dtype=nodevals.dtype)
    d = nodevals.reshape(-1)
    for i in range(len(a)):
        ev[i] = binary_fcn(d[a[i]], d[b[i]])
    return ev


def edges_to_graph(edges, N=-1, weights=None):
    """
    Convert list of pairs of adjacent nodes into a sparse matrix graph representation

    Parameters
    ----------
    edges : 2D array of shape `(2, E)`, or length 2 tuple of 1D arrays of length `E`

        The edges in a graph whose nodes are labelled `0, ..., N-1`.
        Denoting `a = edges[0]` and `b = edges[1]`,
        - `a` and `b` are 1D arrays of type int
        - the nodes labelled `a[i]` and `b[i]` are adjacent, for `0 <= i <= E-1`
        - must have `0 <= a[i], b[i] <= N-1` for `0 <= i <= E-1`, so `edges` is
        referring to valid nodes.

    N : int, Default -1
        Number of nodes in the graph.
        If -1, the number of nodes is assumed to be the largest value in
        `edges`, plus one since elements in `edges` are zero-indexed,
        i.e. nodes are laballed 0, 1, ..., N-1.

    weights : array, Default None
        Weights of the edges in the graph.
        If None, a weight of 1 is given for each edge.

    Returns
    -------
    G : csr_matrix
        Symmetric, sparse, `N` by `N` matrix representation of the undirected
        graph.  `G[m,n] = G[n,m] = weights[i]` where `i` is such that
        `edges[0][i] = m` and `edges[1][i] = n`.
    """
    a, b = edges
    if N < 0:
        N = max(np.max(a), np.max(b)) + 1  # assumes no degree 0 nodes.
    if weights is None:
        weights = np.ones(len(a), dtype=int)
    G = coo_matrix((weights, edges), shape=(N, N))
    return csr_matrix(G + G.T)


@nb.njit
def edges_to_csr(edges, N=-1):
    """
    Convert list of pairs of adjacent nodes into a csr matrix graph
    representation, maintaining the order given in `edges`.

    Parameters
    ----------
    edges, N :
        See `edges_to_graph`.

    Returns
    -------
    indices, indptr : array of int
        Together, these give a CSR representation of the symmetric, sparse,
        `N` by `N` matrix of the undirected graph.
        The neighbour nodes of node `m` are `indices[indptr[m] : indptr[m+1]]`.

    Notes
    -----
    The array `indices[indptr[m] : indptr[m+1]]` not necessarily sorted in
    increasing order, as it is when `indices` and `indptr` are these-named
    attributes in a CSR graph (such as returned by `edges_to_graph`).

    Suppose, for example, that `a, b = edges` is such that `b[i]` is west of
    `a[i]` for the first ~half and `b[i]` is south of `a[i]` for the second
    ~half. Then `indices[indptr[m] : indptr[m+1]]` orders the neighbours of `m`
    as [west, south, north, east] for most nodes `m`, both when `indices` and
    `indptr` are from this function or from `edges_to_graph`. However, when `m`
    is just east of the Prime Meridian so that its western neighbour is just
    west of the Prime Meridian, then `edges_to_graph` gives the neighbours of
    `m` in the order [south, north, east, west] so that these nodes are sorted
    in increasing order. In contrast, this function gives the neighbours of `m`
    in the order [west, south, north, east], consistent with other nodes.
    """
    a, b = edges
    E = len(a)
    if N < 0:  # not using `N is None` for njit
        N = max(np.max(a), np.max(b)) + 1  # assumes no degree 0 nodes.

    # Pre-calculate degree of each node
    deg = np.zeros(N, dtype=np.int_)
    for i in range(E):
        deg[a[i]] += 1
        deg[b[i]] += 1
    sum_deg = np.sum(deg)

    indptr = np.empty(N + 1, dtype=np.int_)
    indices = np.empty(sum_deg, dtype=np.int_)

    # indptr is the cumulative sum of deg, i.e.
    # # indptr[0:-1] = np.cumsum(deg); indptr[-1] = indptr[-2] + deg[-1]
    indptr[0] = 0
    for i in range(N):
        indptr[i + 1] = indptr[i] + deg[i]

    # `ctr[n]` is the index of `indices` at which we'll insert the next
    # neighbour of node `n`. That is, `ctr[n] - indptr[n]` is the current
    # number of nodes we've added as neighbours to node `n`.
    ctr = indptr.copy()

    # Begin accumulating neighbours of each node.
    for i in range(E):
        # Add b[i] as a neighbour to n = a[i]
        n = a[i]
        indices[ctr[n]] = b[i]
        ctr[n] += 1

        # Add a[i] as a neighbour to n = b[i]
        n = b[i]
        indices[ctr[n]] = a[i]
        ctr[n] += 1

    return indptr, indices


def build_grid(G):
    """
    Convert a dict of (undirected) graphs into a grid dict used by neutralocean.

    Parameters
    ----------
    G : matrix or dict
        If a (sparse) matrix, this specifies the edges in the graph:
            an edge exists between nodes `m` and `n`
            iff `G[m,n]` is nonzero or `G[n,m]` is nonzero.
            If `G` is symmetric, only its upper triangle is used.
            If `G` is not symmetric, it must have only one of `G[m,n]` or
            `G[n,m]` be non-zero; this condition is not checked for.
            The geometric distances are both taken to be 1.0, regardless of the
            data in `G`.
        If a dict, must have `'dist'` and `'distperp'` entries, which are both
            (sparse) matrices with the same sparsity structure.
            The edges in the graph are determined from `G['dist']` as above.
            The distance between adjacent nodes `m` and `n` is given by
            `G['dist'][m,n]` or `G['dist'][n,m]`.
            The distance of the interface between adjacent nodes `m` and `n` is
            given by `G['distperp'][m,n]` or `G['distperp'][n,m]`.

    Returns
    -------
    grid : dict

        grid['edges'] : tuple of length 2
            Each element is an array of int of length E, where E is the number of
            edges in the grid's graph, i.e. the number of pairs of adjacent water
            columns (including land) in the grid.
            If `grid['edges'] = (a, b)`, the nodes (water columns) whose linear
            indices are `a[i]` and `b[i]` are adjacent.

        grid['dist'] : 1d array
            `grid['dist'][i]` is the distance between nodes `a[i]` and `b[i]`,
            where `grid['edges'] = (a, b)`.

        grid['distperp'] : 1d array
            `grid['distperp'][i]` is the distance of the interface between
            nodes `a[i]` and `b[i]`, where `grid['edges'] = (a, b)`.
    """

    grid = dict()
    if isinstance(G, dict):
        if not ("dist" in G and "distperp" in G):
            raise ValueError(
                "If a dict, G must have 'dist' and 'distperp' entries"
            )

        D = triu_if_sym(G["dist"])
        P = triu_if_sym(G["distperp"])

        a, b, dist = find(D)
        A, B, perp = find(P)
        if not np.array_equal(a, A) or not np.array_equal(b, B):
            raise ValueError(
                "Given matrices have different sparsity structure."
            )
        grid["edges"] = (a, b)
        grid["dist"] = dist
        grid["distperp"] = perp

    else:  # G is a matrix
        G = triu_if_sym_structure(G)
        a, b, _ = find(G)

        grid["edges"] = (a, b)
        grid["dist"] = grid["distperp"] = np.ones(len(a))

    return grid


def sym_structure(A):
    """
    Determine whether the sparsity structure of martix `A` is symmetric.
    Returns True iff `A[i,j] == 0` implies `A[j,i] == 0` for all valid.
    Also returns False if A is not a square matrix.
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        return False  # A is not a square matrix

    if issparse(A):
        return ((A != 0) != (A.T != 0)).nnz == 0
    else:
        return np.all((A == 0) == (A.T == 0))


def triu_if_sym(A):
    """Return upper triangle matrix if symmetric"""
    if issymmetric(A):
        return triu(A)
    else:
        return A


def triu_if_sym_structure(A):
    """Return upper triangle matrix if symmetric sparsity structure"""
    if sym_structure(A):
        return triu(A)
    else:
        return A
