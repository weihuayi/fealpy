import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.mesh import PolygonMesh as PolygonMesh_old
from fealpy.functionspace.conforming_scalar_ve_space_2d import CVEMDof2d
from fealpy.functionspace.ConformingVirtualElementSpace2d import CVEMDof2d as CVEMDof2d_old



@pytest.mark.parametrize("p", range(1, 10))
def test_dof(p):
    tmesh = TriangleMesh.from_one_triangle()
    tmesh.uniform_refine()
    pmesh = PolygonMesh.from_mesh(tmesh) 
    pmesh_old = PolygonMesh_old.from_mesh(tmesh)
    dof = CVEMDof2d(pmesh,p)
    dof_old = CVEMDof2d_old(pmesh_old,p)
    ips = dof.interpolation_points()
    ldof = dof.number_of_local_dofs()
    assert (ldof==dof_old.number_of_local_dofs()).all()
    print('ldof',ldof)
    gdof = dof.number_of_global_dofs()
    assert gdof == dof_old.number_of_global_dofs()
    print('gdof',gdof)
    isBdDof = dof.is_boundary_dof()
    assert (isBdDof == dof_old.boundary_dof()).all
    edge2dof = dof.edge_to_dof()
    assert (edge2dof == dof_old.edge_to_dof()).all()
    print(edge2dof)
    cell2ipoint,location = dof_old.cell_to_dof()
    cell2dof = dof.cell_to_dof()
    cell2dof_old =  np.hsplit(cell2ipoint, location[1:-1])
    print(cell2dof)
    assert all([(cell2dof[i] == cell2dof_old[i]).all() for i in
        range(len(cell2dof))])

    fig, axes = plt.subplots()
    pmesh.add_plot(axes)
    pmesh.find_node(axes, node =ips, showindex=True)
    pmesh.find_cell(axes, showindex=True)
    pmesh.find_edge(axes, showindex=True)
    plt.show()
if __name__ == "__main__":
    test_dof(3)
