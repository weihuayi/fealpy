
import pytest
import ipdb

import jax
import jax.numpy as jnp

from fealpy.mesh import TriangleMesh as Mesh
from fealpy.jax.mesh import TriangleMesh as TriangleMesh
from fealpy.jax.functionspace import InteriorPenaltyLagrangeFESpace2d
from fealpy.jax import logger
jax.config.update("jax_enable_x64", True)

def test_lagrange_fe_space():

    #mesh = Mesh.from_box(nx=1, ny=1)
    #node = jnp.array(mesh.entity('node'))
    #cell = jnp.array(mesh.entity('cell'))
    node = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    cell = jnp.array([[3, 0, 2], [0, 3, 1]], dtype=jnp.int64)
    print(node)
    print(cell)

    jmesh = TriangleMesh(node, cell)
    edge = jmesh.entity('edge')
    print(edge)

    space = InteriorPenaltyLagrangeFESpace2d(jmesh, p=2)

    bcs = jnp.array([[0.1, 0.2, 0.7]], dtype=jnp.float64)
    ebcs = jnp.array([[0.1, 0.9]], dtype=jnp.float64)

    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs, variable='x')
    jphi1 = space.grad_normal_jump_basis(ebcs)
    jphi2 = space.grad_normal_2_jump_basis(ebcs)

    print("aaa : ", phi.shape)
    print(gphi.shape)
    print(jphi1)
    print(jphi2)




if __name__ == "__main__":
    test_lagrange_fe_space()
