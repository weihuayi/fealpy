
import pytest

import jax
import jax.numpy as jnp

from fealpy.backend import backend_manager as bm

#bm.set_backend('numpy')
bm.set_backend('jax')

node = bm.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]], dtype=bm.float32) # (NN, 2)

edge = bm.array([
     [1,0],
     [0,2],
     [3,0],
     [3,1],
     [2,3]], dtype=bm.int32)

cell = bm.array([
     [2,3,0],
     [1,0,3]], dtype=bm.int32)

cellquad = bm.array([
     [0,1,3,2]], dtype=bm.int32)

bcs = bm.array([
     [0,0,1],
     [1,0,0],
     [0,1,0],
     [1/3,1/3,1/3]], dtype=bm.float32)

bcs1d = bm.array([
     [1/2,1/2],
     [0,1]], dtype=bm.float32)

def test_multi_index_matrix():
    m = bm.multi_index_matrix(3, 2)
    print(m)

def test():
    #re = bm.edge_length(edge, node)
    #re = bm.edge_normal(edge, node, True)
    #re = bm.edge_tangent(edge, node, True)
    #re = bm.barycenter(cell, node)
    #re = bm.simplex_measure(cell, node) 
    #re = bm.triangle_grad_lambda_2d(cell, node) 
    #re = bm.interval_grad_lambda(edge, node) 
    #re = bm.interval_grad_lambda(edge, node) 
    #re = bm.bc_to_points((bcs1d,bcs1d), node, cellquad) 
    #re = bm.bc_to_points(bcs, node, cell) 
    #re = bm._simplex_shape_function_kernel(bcs, 5) 
    #re = bm.simplex_shape_function(jnp.array([bcs,bcs]), 5) 
    #re = bm.simplex_grad_shape_function(bcs, 5) 
    re = bm.simplex_shape_function(jnp.array([bcs,bcs]), 5) 
    print(re)

if __name__ == "__main__":
    #test_multi_index_matrix()
    test()
