import numpy as np
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from conforming_vector_vem_h1_projector import ConformingVectorVEMH1Projector2d 
from fealpy.mesh import PolygonMesh


def test_assembly_cell_righthand_side_and_dof_matrix(p)
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    space =  ConformingVectorVESpace2d(mesh, p=p)
    sspace = ConformingScalarVESpace2d(mesh, p=p-1)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(sspace.smspace) #(n_{k-1}, n_{k-1})
    
    projector = ConformingVectorVEMH1Projector2d(M)
    B = projector.assembly_cell_right_hand_side(space) 
    print(B)
    print(B.shape)

if __name__ == "__main__":
    test_assembly_cell_righthand_side_and_dof_matrix(1)

