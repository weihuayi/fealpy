import pytest
import ipdb

import numpy as np

from fealpy.mesh import TriangleMesh as Mesh
from fealpy.mesh import TriangleMesh as TriangleMesh
from fealpy.functionspace import InteriorPenaltyBernsteinFESpace2d

def test_lagrange_fe_space():

    #mesh = Mesh.from_box(nx=1, ny=1)
    #node = jnp.array(mesh.entity('node'))
    #cell = jnp.array(mesh.entity('cell'))
    node = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    cell = np.array([[2, 3, 0], [3, 1, 0]], dtype=np.int64)
    print(node)
    print(cell)

    mesh = TriangleMesh(node, cell)
    edge = mesh.entity('edge')
    print(edge)

    space = InteriorPenaltyBernsteinFESpace2d(mesh, p=2)

    bcs = np.array([[0.1, 0.2, 0.7]], dtype=np.float64)
    ebcs = np.array([[0.1, 0.9]], dtype=np.float64)

    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    jphi1 = space.grad_normal_jump_basis(ebcs)
    jphi2 = space.grad_normal_2_jump_basis(ebcs)

    print("aaa : ", phi.shape)
    print(gphi.shape)
    print(jphi1)
    print(jphi2)




if __name__ == "__main__":
    test_lagrange_fe_space()
