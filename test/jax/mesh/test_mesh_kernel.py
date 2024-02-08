import pytest
import ipdb

import jax
import jax.numpy as jnp

from fealpy.mesh import TriangleMesh as Mesh

from fealpy.jax.mesh import TriangleMesh as TriangleMesh
from fealpy.jax import logger

from fealpy.jax.mesh.mesh_kernel import *


def test_simplex_shape_function():

    q = 3
    p = 2

    mesh = Mesh.from_box(nx=1, ny=1)
    qf = mesh.integrator(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    TD = mesh.top_dimension()
    mi = mesh.multi_index_matrix(p, TD)
    phi = mesh._shape_function(bcs, p)
    gphi = mesh._grad_shape_function(bcs, p) 

    print(phi)
    print(gphi)


    node = jnp.array(mesh.entity('node'))
    cell = jnp.array(mesh.entity('cell'))
    mesh = TriangleMesh(node, cell)
    qf = mesh.integrator(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    TD = mesh.top_dimension()
    mi = mesh.multi_index_matrix(p, TD)
    phi = simplex_shape_function(bcs, p, mi)
    gphi = grad_simplex_shape_function(bcs, p, mi)
    hphi = hess_simplex_shape_function(bcs, p, mi)

    print(phi)
    print(gphi)
    print(hphi.shape)



if __name__ == "__main__":
    test_simplex_shape_function()



