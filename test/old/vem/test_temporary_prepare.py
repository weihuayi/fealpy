import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from fealpy.vem.temporary_prepare import vector_decomposition, laplace_coefficient, coefficient_of_div_VESpace_represented_by_SMSpace
from fealpy.mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
import ipdb
from fealpy.mesh import UniformMesh2d

def test_coefficient_of_div_VESpace_represented_by_SMSpace(p):
    node = np.array([
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
        dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    
    space =  ConformingVectorVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)
    M = M[:, :p*(p+1)//2, :p*(p+1)//2]
 
    K = coefficient_of_div_VESpace_represented_by_SMSpace(space, M)
    return


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

    degree = 3
    dim = 2
    space = ConformingVectorVESpace2d(mesh, p=degree)
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
def test_laplace_coefficient(p):
    node = np.array([
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
        dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    NC = mesh.number_of_cells()
    space = ConformingVectorVESpace2d(mesh, p)

    E = laplace_coefficient(space, p)
    point = np.array([0,1])
    point = np.tile(point, (NC, 1))
    index = np.arange(NC)
    phi1 = space.vmspace.laplace_basis(point, index=index, p=p)
    phi2 = space.vmspace.basis(point, index=index, p=p-2)
    value = np.einsum('kij,kjl->kil', E, phi2)
    np.testing.assert_equal(phi1, value)








if __name__ == "__main__":
    test_coefficient_of_div_VESpace_represented_by_SMSpace(1)
    test_vector_decomposition(refine=True)
    test_laplace_coefficient(3)
