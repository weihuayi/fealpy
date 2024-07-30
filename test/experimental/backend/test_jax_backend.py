
import pytest

import jax
import jax.numpy as jnp

from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('jax')

node = bm.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]], dtype=bm.float_) # (NN, 2)

edge = bm.array([
     [1,0],
     [0,2],
     [3,0],
     [3,1],
     [2,3]], dtype=bm.int_)

cell = bm.array([
     [2,3,0],
     [1,0,3]], dtype=bm.int_)

def test_multi_index_matrix():
    m = bm.multi_index_matrix(3, 2)
    print(m)

def test():
    #re = bm.edge_length(edge, node)
    #re = bm.edge_normal(edge, node, True)
    #re = bm.edge_tangent(edge, node, True)
    #re = bm.barycenter(cell, node)
    #re = bm.simplex_measure(cell, node)
    print(re)

if __name__ == "__main__":
    #test_multi_index_matrix()
    test()
