import numpy as np
from scipy.sparse import csr_matrix
import pytest

from fealpy.fem import BilinearForm


def test_truss():

    from fealpy.mesh import EdgeMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import TrussStructureIntegrator
    
    mesh = EdgeMesh.from_tower()
    GD = mesh.geo_dimension()
    space = Space(mesh, p=1, doforder='vdims')

    bform = BilinearForm(GD*(space,))
    bform.add_domain_integrator(TrussStructureIntegrator(1500, 2000))
    
    bform.assembly()
    #M = TrussStructureIntegrator(1500,2000)
    #A = M.assembly_cell_matrix(3*(space,))
    #print(bform.M.toarray())
    
    
    GD = mesh.GD
    edge = mesh.entity('edge')
    edge2dof = np.zeros((edge.shape[0], 2*GD), dtype=np.int_)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_cells()
    for i in range(GD):
        edge2dof[:, i::GD] = edge + NN*i
    l = mesh.cell_length().reshape(-1, 1)
    tan = mesh.cell_unit_tangent()
    E = 1500
    A = 2000
    R = np.einsum('ik, im->ikm', tan, tan)
    K = np.zeros((NE, GD*2, GD*2), dtype=np.float64)
    K[:, :GD, :GD] = R
    K[:, -GD:, :GD] = -R
    K[:, :GD, -GD:] = -R
    K[:, -GD:, -GD:] = R
    K *= E*A
    K /= l[:, None]

    I = np.broadcast_to(edge2dof[:, :, None], shape=K.shape)
    J = np.broadcast_to(edge2dof[:, None, :], shape=K.shape)

    K = csr_matrix((K.flat, (I.flat, J.flat)), shape=(NN*GD, NN*GD))
    print(np.sum(np.abs(K.toarray()-bform.M.toarray())))
    print(mesh.number_of_nodes())
    print(K.toarray().shape)

if __name__ == '__main__':
    test_truss()



