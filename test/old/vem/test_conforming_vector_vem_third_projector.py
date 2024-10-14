import numpy as np 
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from fealpy.vem.conforming_vector_vem_l2_projector import ConformingVectorVEML2Projector2d 
from fealpy.vem.conforming_vector_vem_h1_projector import ConformingVectorVEMH1Projector2d 
from fealpy.vem.conforming_vector_vem_third_projector import ConformingVectorVEMThirdProjector2d
from fealpy.vem.conforming_vector_vem_dof_integrator import ConformingVectorVEMDoFIntegrator2d
from fealpy.mesh import PolygonMesh,TriangleMesh


def test_assembly_cell_righthand_side(p):
    node = np.array([(-1.0, -1.0),(1.0, -1.0),(1.0, 1.0),(-1.0,1.0)])
    cell = np.array([0,1,2,3])
    cellLocation = np.array([0,4])
    mesh = PolygonMesh(node, cell, cellLocation)

    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    
    space =  ConformingVectorVESpace2d(mesh, p=p)
    smldof = space.smspace.number_of_local_dofs()
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace, p=p+1) 
    H1projector = ConformingVectorVEMH1Projector2d(M[:, :smldof, :smldof])
    PI1 = H1projector.assembly_cell_matrix(space)
    L2projector = ConformingVectorVEML2Projector2d(M,PI1)
    PI0 = L2projector.assembly_cell_matrix(space)
    third = ConformingVectorVEMThirdProjector2d(M[:, :smldof, :smldof],PI0)
    A = third.assembly_cell_right_hand_side(space)

def test_assembly_cell_left_hand_side(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    node = np.array([(-1.0, -1.0),(1.0, -1.0),(1.0, 1.0),(-1.0,1.0)])
    cell = np.array([0,1,2,3])
    cellLocation = np.array([0,4])
    mesh = PolygonMesh(node, cell, cellLocation)

    space =  ConformingVectorVESpace2d(mesh, p=p)
    smldof = space.smspace.number_of_local_dofs()
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace, p=p+1) 
    H1projector = ConformingVectorVEMH1Projector2d(M[:, :smldof, :smldof])
    PI1 = H1projector.assembly_cell_matrix(space)
    L2projector = ConformingVectorVEML2Projector2d(M,PI1)
    PI0 = L2projector.assembly_cell_matrix(space)
    third = ConformingVectorVEMThirdProjector2d(M[:, :smldof, :smldof],PI0)
    F = third.assembly_cell_left_hand_side(space)


def test_assembly_cell_matrix(p):
    node = np.array([(-1.0, -1.0),(1.0, -1.0),(1.0, 1.0),(-1.0,1.0)])
    cell = np.array([0,1,2,3])
    cellLocation = np.array([0,4])
    mesh = PolygonMesh(node, cell, cellLocation)

    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    mesh = PolygonMesh.distorted_concave_rhombic_quadrilaterals_mesh()
    #mesh = PolygonMesh.nonconvex_octagonal_mesh()

    h = np.sqrt(mesh.cell_area())
    NC = mesh.number_of_cells()

    space =  ConformingVectorVESpace2d(mesh, p=p)
    smldof = space.smspace.number_of_local_dofs()
 
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace, p=p+1) 
    H1projector = ConformingVectorVEMH1Projector2d(M[:, :smldof, :smldof])
    PI1 = H1projector.assembly_cell_matrix(space)
    L2projector = ConformingVectorVEML2Projector2d(M,PI1)
    PI0 = L2projector.assembly_cell_matrix(space)
    third = ConformingVectorVEMThirdProjector2d(M[:, :smldof, :smldof],PI0)
    matrix = third.assembly_cell_matrix(space)

    dof = ConformingVectorVEMDoFIntegrator2d()
    D = dof.assembly_cell_matrix(space)

    data = space.smspace.diff_index_1(p=p)
    x = data['x']
    y = data['y']
    smldof1 = p*(p+1)//2
    for i in range(NC):
        K = np.zeros((2*p*(p+1), 2*smldof))
        K[np.arange(smldof1),x[0]] = x[1]/h[i]
        K[np.arange(smldof1)+2*smldof1, y[0]] = y[1]/h[i]
        K[np.arange(smldof1)+smldof1, x[0]+smldof] = x[1]/h[i]
        K[np.arange(smldof1)+smldof1*3, y[0]+smldof]= y[1]/h[i]
        val = matrix[i]@D[i]
        np.testing.assert_allclose(K, val, atol=1e-10)
        


if __name__ == "__main__":
    #test_assembly_cell_righthand_side(2)
    #test_assembly_cell_left_hand_side(2)
    test_assembly_cell_matrix(3)
