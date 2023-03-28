import numpy as np
import pytest

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh

@pytest.mark.parametrize("p", range(1, 10))
def test_interval_mesh(p):
    mesh = IntervalMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() ==  (p+1)*(p+2)/2
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

    mesh = TriangleMesh.from_unit_square(nx=2, ny=2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() == NN + (p-1)*NE + (p-2)*(p-1)/2*NC  
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

@pytest.mark.parametrize("p", range(1, 10))
def test_triangle_mesh(p):
    mesh = TriangleMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() ==  (p+1)*(p+2)/2
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

    mesh = TriangleMesh.from_unit_square(nx=2, ny=2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() == NN + (p-1)*NE + (p-2)*(p-1)/2*NC  
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

@pytest.mark.parametrize("p", range(10))
def test_tetrahedron_mesh(p):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() ==  (p+3)*(p+2)*(p+1)//6
    assert space.number_of_local_dofs() == (p+3)*(p+2)*(p+1)//6

    mesh = TetrahedronMesh.from_unit_cube(nx=10, ny=10, nz=10)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() == NN + NE*(p-1) + NF*(p-2)*(p-1)//2 + NC*(p-3)*(p-2)*(p-1)//6
    assert space.number_of_local_dofs() == (p+1)*(p+2)*(p+3)//6

    mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=0.2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() == NN + NE*(p-1) + NF*(p-2)*(p-1)//2 + NC*(p-3)*(p-2)*(p-1)//6
    assert space.number_of_local_dofs() == (p+1)*(p+2)*(p+3)//6



if __name__ == '__main__':
    test_triangle_mesh(2)

