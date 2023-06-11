import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace import ConformingVirtualElementSpace2d
import ipdb
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem import ConformingScalarVEMLaplaceIntegrator2d
def test_assembly_cell_righthand_side_and_dof_matrix(p,plot=False):
    nx = 2
    ny = 2
    dim = 2
    domain = np.array([0, 1, 0, 1])
    mesh = MF.boxmesh2d(domain, nx, ny, meshtype ='poly')
    space = ConformingVirtualElementSpace2d(mesh, p=p)
    realB = space.matrix_B()
    H = space.H
    realD = space.matrix_D(H)
    realG = space.matrix_G(realB, realD)
    realPI1 = space.matrix_PI_1(realG,realB)

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


    ldof = mesh.number_of_local_ipoints(p)
    a = np.add.accumulate(ldof)
    NC = mesh.number_of_cells()
    location = np.zeros(NC+1, dtype=np.int_)
    location[1:] = a
    for i in range(NC):
        if p==2 or p==3:
            np.testing.assert_equal(realB[:,location[i]:location[i+1]], B[i])
            np.testing.assert_equal(realD[location[i]:location[i+1],:], D[i])
            #np.testing.assert_equal(realG[i], G[i]) 
            np.testing.assert_equal(realPI1[i], PI1[i])
        else:
            np.testing.assert_allclose(realB[:,location[i]:location[i+1]], B[i],atol=1e-14)
            np.testing.assert_allclose(realD[location[i]:location[i+1],:], D[i], atol=1e-14)
            #np.testing.assert_allclose(realG[i], G[i] ,atol=1e-10)
            np.testing.assert_allclose(realPI1[i], PI1[i], atol=1e-10)
        i = i + 1


            

    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()
if __name__ == "__main__":
    test_assembly_cell_righthand_side_and_dof_matrix(3)

