from fealpy.backend import backend_manager as bm
from scipy.spatial import cKDTree

class Neighbor:
    def __init__(self, mesh):
        self.mesh = mesh 

    def find_neighbors(state, h):
        tree = cKDTree(state["position"])
        neighbors = tree.query_ball_tree(tree, h)
        n_long = bm.array([len(sublist) for sublist in neighbors])

        neighbors = bm.array([item for sublist in neighbors for item in sublist])

        indices = bm.arange(len(n_long))
        indices = bm.repeat(indices, n_long)
        
        return neighbors, indices
