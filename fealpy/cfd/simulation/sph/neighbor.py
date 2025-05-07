from fealpy.backend import backend_manager as bm
from scipy.spatial import cKDTree

import jax.numpy as jnp
from jax_md import space, partition
from jax import vmap, lax

class Neighbor:
    def __init__(self, mesh):
        self.mesh = mesh 
    
    def segmensum(self, ):
        pass


    def find_neighbors_backend(self, state, h):
        tree = cKDTree(state["position"])
        neighbors = tree.query_ball_tree(tree, h)
        n_long = bm.array([len(sublist) for sublist in neighbors])

        neighbors = bm.array([item for sublist in neighbors for item in sublist])

        indices = bm.arange(len(n_long))
        indices = bm.repeat(indices, n_long)
        
        return neighbors, indices

    def find_neighbors_jax(self, box_size, h):
        '''
        @brief Find neighbor particles within the smoothing radius.

        note : Currently using jax's own jax_md, vmap, lax
        '''
        displacement, shift = space.periodic(box_size)
        neighbor_fn = partition.neighbor_list(displacement, box_size, h)

        nbrs = neighbor_fn.allocate(self.node)
        nbrs = neighbor_fn.update(self.node, nbrs)
        neighbor = nbrs.idx
        num = self.node.shape[0]
        index = vmap(lambda idx, row: jnp.hstack([row, jnp.array([idx])]))(bm.arange(neighbor.shape[0]), neighbor)
        row_len = bm.sum(index != num,axis=1)
        indptr = lax.scan(lambda carry, x: (carry + x, carry + x), 0, row_len)[1]
        indptr = bm.concatenate((bm.tensor([0]), indptr))
        index = index[index != num]

        return index, indptr

