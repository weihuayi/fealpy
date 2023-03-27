import numpy as np
import pytest

from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh

@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
def test_triangle_mesh(p):
    mesh = TriangleMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() ==  (p+1)*(p+2)/2
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

    mesh = TriangleMesh.from_unit_square(nx=10, ny=10)
    space = LagrangeFESpace(mesh, p=p)
    NC = mesh.number_of_cells()
    NE = mesh.number_of_edges()
    NN = mesh.number_of_nodes()
    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() == NN + (p-1)*NE + (p-1)*p/2*NC  
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2




if __name__ == '__main__':
    test_triangle_mesh()

