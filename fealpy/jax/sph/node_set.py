
import jax.numpy as jnp
from scipy.spatial import cKDTree 
import jax.random
class NodeSet():

    def __init__(self, NN, node=None, h=None):
        self.NN = NN
        self.nodedata['node'] = node
        self.h = h


    def number_of_node(self):
        return self.NN 
       
    def interpolate(self, u, node, dim=1, kernel=1):
        node = self.nodedata['node']
    
    def find_neighbors(self):
        tree = cKDTree(self.node)
        neighbors = tree.query_ball_tree(tree,self.h)
        return neighbors
