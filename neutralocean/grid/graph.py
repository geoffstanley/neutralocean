import numpy as np
import numba as nb

from scipy.sparse import csr_matrix, triu


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
def edges_binary_fcn(a, edges, binary_fcn):
    """
    Parameters
    ----------
    a : ndarray
        values at the nodes in a graph

    edges : 2d array of int
        Indices to pairs of nodes in the graph, which define an edge in the graph.
        That is, if `edges[i,0] == m` and `edges[i,1] == n`, then the nodes
        labelled m and n are adjacent.  If one of `edges[i,:]` is less than 0,
        that edge does not exist.

    binary_fcn : function
        A `numba.njit`ed function accepting two numbers and returning one number.

    Returns
    -------
    ev : 1d array
        The value of `binary_fcn` evaluated on each edge.
        That is, `ev[i] = binary_fcn(a[m], a[n])` where `m = edges[i,0]` and
        `n = edges[i,1]`.  If either `m` or `n` is < 0, `ev[i]` is nan.
    """
    ev = np.empty(edges.shape[0], dtype=a.dtype)
    a_ = a.reshape(-1)
    for e in range(len(ev)):
        m, n = edges[e]
        if m >= 0 and n >= 0:
            ev[e] = binary_fcn(a_[m], a_[n])
        else:
            ev[e] = np.nan
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
    E = edges.shape[0]
    if N is None:
        N = np.max(edges) + 1  # assumes no degree 0 nodes.
    r = np.concatenate((edges[:, 0], edges[:, 1]))
    c = np.concatenate((edges[:, 1], edges[:, 0]))
    if weights is None:
        v = np.ones(E * 2, dtype=int)
    else:
        v = np.tile(weights, 2)
    graph = csr_matrix((v, (r, c)), shape=(N, N))
    return graph


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

    Notes
    -----
    Edges with an associated data value of 0 are not supported, as this would
    not be stored in the sparse matrix.

    """

    return triu(graph).tocoo().data
