import numpy as np

def is_separating_axis(n, P1, P2):
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')
    for v in P1:
        proj = np.dot(v, n)
        min1 = min(min1, proj)
        max1 = max(max1, proj)
    for v in P2:
        proj = np.dot(v, n)
        min2 = min(min2, proj)
        max2 = max(max2, proj)
    
    if max1 >= min2 and max2 >= min1:
        return False
    
    return True

def find_edges_norms(P):
    edges = []
    norms = []
    num_edge = len(P)
    for i in range(num_edge):
        edge = P[(i + 1)%num_edge] - P[i]
        norm = np.array([-edge[1], edge[0]])
        edges.append(edge)
        norms.append(norm)
    return edges, norms
        
def collide(P1, P2):
    """
    Check if two polygons overlap.
    We follow https://hackmd.io/@US4ofdv7Sq2GRdxti381_A/ryFmIZrsl?type=view
    Args:
        p1 (List): List of the vertices of a polygon
        p2 (List): List of the vertices of a polygon
    """
    
    P1 = [np.array(v, 'float64') for v in P1]
    P2 = [np.array(v, 'float64') for v in P2]
    
    _, norms1 = find_edges_norms(P1)
    _, norms2 = find_edges_norms(P2)
    norms = norms1 + norms2
    for n in norms:
        if is_separating_axis(n, P1, P2):
            return False
        
    return True