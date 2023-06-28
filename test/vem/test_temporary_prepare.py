import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from fealpy.vem.temporary_prepare import vector_decomposition
from fealpy.mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
import ipdb
from fealpy.mesh import UniformMesh2d




def test_vector_decomposition(refine=False):
    if refine:
        node = np.array([
            (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (0.0, 0.0), 
            (0.0, 1.0), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)], dtype=np.float64)
        cell = np.array([0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8,
            5],dtype=np.int_)
        cellLocation = np.array([0, 4, 8, 12, 16])
        mesh = PolygonMesh(node, cell, cellLocation)
    else:
        node = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0]],dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
        mesh = PolygonMesh(node, cell)

    degree = 2
    dim = 2
    space =  ConformingScalarVESpace2d(mesh, p=degree)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)
    NC = mesh.number_of_cells()
    ipoint = np.array([1,1])
    ipoint = np.tile(ipoint, (NC, 1))

    phi = space.smspace.basis(ipoint)
    sldof = space.smspace.number_of_local_dofs()

    for i in range(NC):
        A = np.zeros((2*sldof, 2))
        A[:sldof, 0] =  phi[i, :]
        A[sldof:sldof*2, 1] = phi[i, :]

        gphi = space.smspace.grad_basis(ipoint, p=degree+1)[i, :, :]
        phi2 = space.smspace.basis(ipoint, p=degree-1)[i, :]
        x = np.array([phi[i][2], -phi[i][1]])
        B = np.einsum('i,j->ij', phi2, x)
        a, b = vector_decomposition(space,p=degree)

        C = np.einsum('ij, jm -> im', a[i], gphi)
        B = np.einsum('ij, jk ->ik', b, B)
        value = C +B
        np.testing.assert_equal(A, value)

if __name__ == "__main__":
    test_vector_decomposition(refine=True)
