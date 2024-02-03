import pytest

import jax
import jax.numpy as jnp

from fealpy.mesh import TriangleMesh
from fealpy.jax.mesh import TriangleMesh as JaxTriangleMesh
from fealpy.jax import logger


def test_cell_area():
    mesh = TriangleMesh.from_box(nx=1, ny=1)
    node = jnp.array(mesh.entity('node'))
    cell = jnp.array(mesh.entity('cell'))

    jmesh = JaxTriangleMesh(node, cell)

    a0 = jmesh.cell_area()
    a1, jac = jmesh.cell_area_with_jac()

    print(a0)
    print(a1)
    print(jac)



if __name__ == "__main__":
    test_cell_area()



