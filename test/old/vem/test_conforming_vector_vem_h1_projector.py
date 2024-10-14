import numpy as np
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from fealpy.mesh import TriangleMesh, PolygonMesh
from fealpy.vem.conforming_vector_vem_h1_projector import ConformingVectorVEMH1Projector2d 
from fealpy.quadrature import GaussLobattoQuadrature
from fealpy.vem.conforming_vector_vem_dof_integrator import ConformingVectorVEMDoFIntegrator2d
import ipdb
def test_assembly_cell_righthand_side(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    
    a = mesh.entity_barycenter()
    nm = mesh.edge_normal()
    h = np.sqrt(mesh.cell_area())
    qf = GaussLobattoQuadrature(p+1)
    bcs, ws = qf.quadpts, qf.weights
    space =  ConformingVectorVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace) #(n_{k-1}, n_{k-1})
    projector = ConformingVectorVEMH1Projector2d(M)
    ipdb.set_trace()
    B = projector.assembly_cell_right_hand_side(space) 
def test_assembly_cell_left_hand_side(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    bc = mesh.entity_barycenter()
    h = np.sqrt(mesh.cell_area())

    space =  ConformingVectorVESpace2d(mesh, p=p)
    sspace = ConformingScalarVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    #ipdb.set_trace()
    M = m.assembly_cell_matrix(sspace.smspace) 
    projector = ConformingVectorVEMH1Projector2d(M)
    
    G = projector.assembly_cell_left_hand_side(space) 
def test_assembly_cell_matrix(p):
    nx = 10
    ny = 10
    domain = np.array([0, 1, 0, 1])
    tmesh = TriangleMesh.from_box()
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
    mesh = PolygonMesh.distorted_concave_rhombic_quadrilaterals_mesh()
    mesh = PolygonMesh.nonconvex_octagonal_mesh()




    space =  ConformingVectorVESpace2d(mesh, p=p)
    sspace = ConformingScalarVESpace2d(mesh, p=p)
    smldof = space.smspace.number_of_local_dofs()
    dof = ConformingVectorVEMDoFIntegrator2d()
    dof_matrix = dof.assembly_cell_matrix(space)
    #print('dof',dof_matrix)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace, p=p+1) 
    H1projector = ConformingVectorVEMH1Projector2d(M[:,:smldof,:smldof])
    G = H1projector.assembly_cell_left_hand_side(space)
    #print('G',G)
    B = H1projector.assembly_cell_right_hand_side(space)
    #print('B',B)
    PI1 = H1projector.assembly_cell_matrix(space)    
    #print('PI1',PI1)

    NC = mesh.number_of_cells()
    for i in range(NC):
        val = PI1[i]@dof_matrix[i]
        #print('val',val)
        np.testing.assert_allclose(val, np.eye(val.shape[0]),atol=1e-12)


if __name__ == "__main__":
    test_assembly_cell_righthand_side(4)
    #test_assembly_cell_left_hand_side(3)
    test_assembly_cell_matrix(1)
