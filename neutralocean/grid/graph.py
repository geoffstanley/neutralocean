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
    edges : tuple
        The edges in a graph, a 2-element tuple with the following:

        a, b : array of int
            Pairs of adjacent nodes in the graph.
            Nodes `a[i]` and `b[i]`, are adjacent.

    nodevals : array-like
        values at the nodes in a graph

    binary_fcn : function
        A `numba.njit` function accepting two numbers and returning one number.

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


def edges_to_graph(edges, N=None, weights=None):
    """
    Convert list of pairs of adjacent nodes into a sparse matrix graph representation

    Parameters
    ----------
    edges : tuple
        Each element is an array of int of length E, where E is the number of
        edges in the grid's graph, i.e. the number of pairs of adjacent water
        columns (including land) in the grid.
        If `edges = (a, b)`, the nodes (water columns) whose linear indices are
        `a[i]` and `b[i]` are adjacent.

    N : int, Default None
        Number of nodes in the graph.
        If None, the number of nodes is assumed to be the largest value in
        `edges`, plus one since elements in `edges` are zero-indexed, i.e. nodes
        are laballed 0, 1, ..., N-1.

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
    E = len(a)
    if N is None:
        N = np.max((np.max(a), np.max(b))) + 1  # assumes no degree 0 nodes.
    if weights is None:
        weights = np.ones(E, dtype=int)
    G = coo_matrix((weights, edges), shape=(N, N))
    return csr_matrix(G + G.T)


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
