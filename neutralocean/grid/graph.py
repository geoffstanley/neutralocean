import numpy as np
import numba as nb

from scipy.sparse import coo_matrix, csr_matrix, find, triu


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
            If one of `a[i]` or `b[i]` is less than 0, that edge does not exist. # TODO:  Is this necessary?

    nodevals : array-like
        values at the nodes in a graph

    binary_fcn : function
        A `numba.njit`ed function accepting two numbers and returning one number.

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
    for i in range(len(ev)):
        m = a[i]
        n = b[i]
        if m >= 0 and n >= 0:  # TODO:  Is this necessary?
            ev[i] = binary_fcn(d[m], d[n])
        else:
            ev[i] = np.nan
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


def build_grid(Gs):
    """
    Convert a dict of (undirected) graphs into a grid dict used by neutralocean.

    Parameters
    ----------
    Gs : dict
        Dictionary of (sparse) matrices, each with the same sparsity structure.
        Letting `G` be one of elements of `Gs`, the associated undirected graph
        has an edge between nodes `m` and `n` iff `G[m,n]` is nonzero or
        `G[m,n]` is nonzero.  If `G` is symmetric or anti-symmetric, only its
        upper triangle is used.  Should have 'dist' and 'distperp' entries.

    Returns
    -------
    grid : dict

        grid['edges'] : tuple of length 2
            Each element is an array of int of length E, where E is the number of
            edges in the grid's graph, i.e. the number of pairs of adjacent water
            columns (including land) in the grid.
            If `edges = (a, b)`, the nodes (water columns) whose linear indices are
            `a[i]` and `b[i]` are adjacent.

        grid[key] : 1d array
            For each key in `Gs`, the data from Gs[key], in the same order as
            grid['edges'].  That is, `grid[key][i]` is the value of `Gs[key]`
            at entry `(grid['edges'][0][i], grid['edges'][1][i])`.

    """

    grid = dict()
    first = True
    for key in Gs:
        G = Gs[key]
        if np.max(abs(G - G.T)) == 0 or np.max(abs(G + G.T)) == 0:
            # symmetric or anti-symmetric
            G = triu(G)
        a_, b_, data = find(G)
        if first:
            a, b = a_, b_
            grid["edges"] = (a, b)
            first = False
        else:
            if not np.array_equal(a, a_) or not np.array_equal(b, b_):
                raise ValueError(
                    "Given matrices have different sparsity structure."
                )
        grid[key] = data

    return grid
