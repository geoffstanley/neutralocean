import numpy as np
import numba as nb

from scipy.sparse import coo_matrix, csr_matrix, find, triu


@nb.njit
def shift(a, adj):
    """
    Parameters
    ----------
    a : ndarray
        input data

    adj : 1d array
        linear indices to elements of `a`

    Returns
    -------
    b : ndarray
        The i'th element of b (in linear order) is the adj[i]'th element of `a`,
        (in linear order), or nan if adj[i] is < 0.  Same shape as a.
    """
    b = np.empty(a.size, dtype=a.dtype)
    sh = a.shape
    a_ = a.reshape(-1)
    for i in range(len(b)):
        if adj[i] >= 0:
            b[i] = a_[adj[i]]
        else:
            b[i] = np.nan
    return b.reshape(sh)


@nb.njit
def graph_binary_fcn(G, nodevals, binary_fcn):
    """
    Apply a binary function to all pairs of data for all pairs of adjacent nodes
    in a graph specified by a sparse COO matrix.

    Parameters
    ----------
    G : tuple
        The graph, a 2-element tuple:

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

    # if isinstance(G, tuple) and len(G) == 2:
    a, b = G
    # elif scipy.sparse.issparse(G):
    #     G = G.tocoo()
    #     a = G.row
    #     b = G.col
    # else:
    #     raise ValueError(
    #         "G should be a tuple with 2 elements or a sparse matrix."
    #     )

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


@nb.njit
def max_deg_from_edges(edges, n_nodes):
    deg = np.zeros((n_nodes,), dtype=type(0))
    for e in range(edges.shape[0]):
        a, b = edges[e]
        if a >= 0 and b >= 0:
            deg[a] += 1
            deg[b] += 1
    return np.max(deg)


@nb.njit
def edges_to_adjnodes(edges, n_nodes, max_deg):
    """
    Convert list of pairs of nodes into list of nodes adjacent to each node.

    Parameters
    ----------
    edges : ndarray of int
        Indices to pairs of nodes in the graph, which define an edge in the graph.
        That is, if `edges[i,0] == m` and `edges[i,1] == n`, then the nodes
        labelled m and n are adjacent.  If one of `edges[i,:]` is less than 0,
        that edge does not exist.

    n_nodes : int
        Number of nodes in the graph

    max_deg : int
        Maximum degree (number of adjacent nodes) in the graph.

    Returns
    -------
    adjnodes : ndarray of int
        2D array of size `(n_nodes, max_degree)` giving indices to the nodes
        adjacent to each node.  That is, `adjnodes[`i`,:]` gives indices to
        all nodes adjacent to node `m`.  When node `m` has fewer than `max_deg`
        neighbours, some elements of `adjnodes[m,:]` will be -1.

    """
    # adjnodes = np.full((n_nodes, max_deg), -1)
    adjnodes = np.full((n_nodes, max_deg), n_nodes)
    deg = np.zeros(n_nodes, dtype=type(0))
    for e in range(edges.shape[0]):
        a, b = edges[e]
        if a >= 0 and b >= 0:  # TODO: needed?
            adjnodes[a, deg[a]] = b
            adjnodes[b, deg[b]] = a
            deg[a] += 1
            deg[b] += 1
    return adjnodes


@nb.njit
def edgescompact_to_adjnodes(adj):
    # TODO:  Just make a sprase matrix, symmetric, then convert to csr_matrix, then return indptrs and indices.

    # assumes a reasonable layout of adj

    n_nodes = adj.shape[0]
    D = adj.shape[1]

    adjnodes = np.hstack((adj, np.full((n_nodes, D), -1)))
    deg = np.full((n_nodes,), D)
    for m in range(n_nodes):
        for d in range(D):
            n = adj[m, d]  # neighbour node
            if n >= 0:
                # n is adjacent to m.
                # Make symmetric link: m is adjacent to n
                adjnodes[n, deg[n]] = m
                deg[n] += 1
    return adjnodes


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


def graph_to_edges(graph):
    """
    Build a list of pairs of nodes that are adjacent in a graph

    Parameters
    ----------
    graph : sparse matrix
        A sparse N x N matrix representing a graph (collection of nodes and
        edges), with N nodes and an edge between nodes `i` and `j` iff
        `graph[i,j]` is non-zero.
        Note: Only the upper triangle of the matrix is used, as the graph
        is assumed to be undirected.

    Returns
    -------
    edges : ndarray of int
        An array of pairs of nodes that are adjacent.
        The shape of edges is (E,2) where `E` is the number of edges in the graph.
        Nodes numbered by `edges[i,0]` and `edges[i,1]` are adjacent.
    """
    rcv = triu(graph).tocoo()
    return np.stack((rcv.row, rcv.col), axis=1)


def graph_to_edge_data(graph):
    """
    Build a list of data on the edges

    Parameters
    ----------
    graph : sparse matrix
        As in `graph_to_edges`.

    Returns
    -------
    edge_data : array
        A 1D array of weights or values of the edges, listed in the same order
        as `edges` returned by `graph_to_edges`.
        That is, `edge_data[m] = graph[i,j]` is the value associated with the
        edge between nodes `i` and `j`, where `i = edges[m,0]` and `j = edges[m,1]`.

    """

    return triu(graph).tocoo().data


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


@nb.njit
def intersperse(a, b):
    # Take two arrays of equal length and create an array twice as long,
    # having the elements of the two arrays interspersed in alternating order.
    # This is slightly faster than >> c = np.stack((a,b), axis=-1).reshape(-1)
    c = np.empty(2 * len(a), dtype=a.dtype)
    for i in range(len(a)):
        c[2 * i] = a[i]
        c[2 * i + 1] = b[i]
    return c
