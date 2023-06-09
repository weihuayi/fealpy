import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem  import ConformingScalarVEML2Projector2d

def test_assembly_cell_righthand_side_and_dof_matrix(p,plot=False):
    nx = 10
    ny = 10
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

    tmesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
    space =  ConformingScalarVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = ConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)

 

    projector = ConformingScalarVEMH1Projector2d(D)
    B = projector.assembly_cell_right_hand_side(space)
    PI1 = projector.assembly_cell_matrix(space)


    a = ConformingScalarVEML2Projector2d(M, PI1)
    C = a.assembly_cell_right_hand_side(space)
    PI0 = a.assembly_cell_matrix(space)

    NC = mesh.number_of_cells()
    for i in range(NC):
        np.testing.assert_allclose(realC[i], C[i] ,atol=1e-10)
        np.testing.assert_allclose(realPI0[i], PI0[i], atol=1e-10)
        if p==2 or p ==3 or p==1:
            np.testing.assert_equal(realC[i], C[i])
            np.testing.assert_equal(realPI0[i], PI0[i])
        else:
            np.testing.assert_allclose(realC[i], C[i] ,atol=1e-10)
            np.testing.assert_allclose(realPI0[i], PI0[i], atol=1e-10)
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

