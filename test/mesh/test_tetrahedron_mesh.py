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

def test_mesh_generation_on_cylinder_by_gmsh():
    mesh = TetrahedronMesh.from_cylinder_gmsh(1, 5, 0.1)
    mesh.add_plot(plt)
    plt.show()

def test_mesh_generation_by_meshpy():
    points = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ])

    facets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    
    mesh = TetrahedronMesh.from_meshpy(points, facets, 0.2)
    mesh.add_plot(plt)
    plt.show()


if __name__ == "__main__":
    test_mesh_generation_by_meshpy()
