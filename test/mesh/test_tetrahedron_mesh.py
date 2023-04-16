import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.mesh import TetrahedronMesh 
from fealpy.functionspace import LagrangeFiniteElementSpace

import ipdb

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_uniform_refine(n):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    mesh.uniform_refine(n=n)

    vol = mesh.entity_measure('cell')
    assert np.all(vol>0)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_interpolate(p):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')

    mesh.uniform_refine(n=3)

    ips0 = mesh.interpolation_points(p)

    space = LagrangeFiniteElementSpace(mesh, p=p)
    ips1 = space.interpolation_points()

    assert np.allclose(ips0, ips1)


    c2d0 = mesh.cell_to_ipoint(p)
    c2d1 = space.cell_to_dof()

    assert np.all(c2d0 == c2d1)



if __name__ == "__main__":
    test_interpolate(1)
