import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.functionspace.conforming_vector_ve_space_2d import CNSVEMDof2d
from fealpy.functionspace.conforming_scalar_ve_space_2d import CVEMDof2d

@pytest.mark.parametrize("p", range(1, 10))
def test_dof(p):
    tmesh = TriangleMesh.from_one_triangle()
    tmesh.uniform_refine()
    pmesh = PolygonMesh.from_mesh(tmesh) 
    dof = CNSVEMDof2d(pmesh, p)
    isBdDof = dof.is_boundary_dof()
    print(isBdDof)
    dof1 = CVEMDof2d(pmesh, p)
    isBdDof1 = dof1.is_boundary_dof()
    print(isBdDof1)
    gdof = dof.number_of_global_dofs()
    gdof1 = dof1.number_of_global_dofs()
    print(gdof)
    print(gdof1)
    local = dof.number_of_local_dofs()
    print(local)
    edge2dof = dof.edge_to_dof()
    edge2dof1 = dof1.edge_to_dof()
    print(edge2dof,edge2dof1)
    cell2dof = dof.cell_to_dof()
    cell2dof1 = dof1.cell_to_dof()
    print(cell2dof,cell2dof1)

    fig, axes = plt.subplots()
    pmesh.add_plot(axes)
    pmesh.find_node(axes, showindex=True)
    pmesh.find_cell(axes, showindex=True)
    pmesh.find_edge(axes, showindex=True)
    plt.show()
if __name__ == "__main__":
    test_dof(3)
