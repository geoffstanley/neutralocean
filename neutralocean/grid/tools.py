from neutralocean.lib import aggsum


def divergence(vector, edges, distperp=1.0, area=1.0):
    """
    Calculate the divergence of a vector field defined on the edges of a graph

    `vector[i]` is positive when the vector field points from node `a[i]` to
    node `b[i]`, where `(a,b) = edges`.
    `distperp[i]` is the distance of the interface between nodes `a[i]` and `b[i]`.
    `area[i]` is the area of node `i`.
    
    Note: This function's API is likely to change, to just take in a `grid` dict.
    """

    a, b = edges
    N = max(max(a), max(b)) + 1  # number of nodes
    e = vector * distperp
    divg = (aggsum(e, a, N) - aggsum(e, b, N)) / area
    return divg
