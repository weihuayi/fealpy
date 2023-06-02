import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.vem.conforming_scalar_vem_h1_projector import ConformingScalarVEMH1Projector2d
from fealpy.vem.conforming_scalar_vem_l2_projector import ConformingScalarVEML2Projector2d
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.polygon_mesh import PolygonMesh

def test_assembly_cell_righthand_side_and_dof_matrix(p,plot=False):
    nx = 2
    ny = 2
    domain = np.array([0, 1, 0, 1])
    mesh = MF.boxmesh2d(domain, nx, ny, meshtype ='poly')
    space = ConformingVirtualElementSpace2d(mesh, p=p)
    realB = space.matrix_B()
    H = space.H
    realD = space.matrix_D(H)
    realG = space.matrix_G(realB, realD)
    realPI1 = space.matrix_PI_1(realG,realB)
    realC = space.matrix_C(H,realPI1)
    realPI0 = space.matrix_PI_0(H,realC)

    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()


    mesh = MF.boxmesh2d(domain, nx, ny, meshtype ='tri')
    mesh = PolygonMesh.from_triangle_mesh_by_dual(mesh)
    space =  ConformingScalarVESpace2d(mesh, p=p)
    b = ConformingScalarVEML2Projector2d()
    C = b.assembly_cell_righthand_side(space)
    PI0 = b.assembly_cell_matrix(space)


    NC = mesh.number_of_cells()
    for i in range(NC):
        np.testing.assert_allclose(realC[i], C[i] ,atol=1e-10)
        np.testing.assert_allclose(realPI0[i], PI0[i], atol=1e-10)

        #np.testing.assert_equal(realC[i], C[i])
        #np.testing.assert_equal(realPI0[i], PI0[i])

        i = i+1

    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()
if __name__ == "__main__":
    test_assembly_cell_righthand_side_and_dof_matrix(1)

