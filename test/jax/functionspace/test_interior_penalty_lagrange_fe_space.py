
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

    mesh = Mesh.from_box(nx=1, ny=1)
    node = jnp.array(mesh.entity('node'))
    cell = jnp.array(mesh.entity('cell'))

    jmesh = TriangleMesh(node, cell)

    space = InteriorPenaltyLagrangeFESpace2d(jmesh, p=1)

    bcs = jnp.array([[0.1, 0.2, 0.7]], dtype=jnp.float64)

    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)

    print(phi)
    print(gphi)




if __name__ == "__main__":
    test_lagrange_fe_space()
