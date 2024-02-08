import pytest
import ipdb

import jax
import jax.numpy as jnp

from fealpy.mesh import TriangleMesh as Mesh
from fealpy.jax.mesh import TriangleMesh as TriangleMesh
from fealpy.jax import logger


def test_cell_area():
    mesh = Mesh.from_box(nx=1, ny=1)
    node = jnp.array(mesh.entity('node'))
    cell = jnp.array(mesh.entity('cell'))

    mesh = TriangleMesh(node, cell)

    a0 = jmesh.cell_area()
    a1, jac = mesh.cell_area_with_jac()

    print(a0)
    print(a1)
    print(jac)
    print(mesh.ds.edge)
    print(mesh.ds.edge2cell)



if __name__ == "__main__":
    test_cell_area()



