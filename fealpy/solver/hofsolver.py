import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import spsolve, LinearOperator
import pyamg
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..femmodel.doperator import stiff_matrix


class HOFEMFastSovler():
    def __init__(self, A, space, integrator, measure):
        self.DL = tril(A).tocsr()
        self.U = triu(A, k=1).tocsr()

        self.DU = tiru(A).tocsr()
        self.L =  tril(A, k=-1).tocsr()

        linspace = LagrangeFiniteElementSpace(space.mesh, 1)

        # construct amg solver 
        A1 = stiff_matrix(linspace, integrator, measure)
        isBdDof = linspace.boundary_dof()
        bdIdx = np.zeros((A1.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A1.shape[0], A1.shape[0])
        T = spdiags(1-bdIdx, 0, A1.shape[0], A1.shape[0])
        A1 = T@A@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(A1)  

        # Get interpolation matrix 
        NC = space.mesh.number_of_cells()
        bc = space.dof.multiIndex/space.p
        val = np.tile(bc, (NC, 1))
        c2d0 = space.cell_to_dof()
        c2d1 = linspace.cell_to_dof()

        I = np.einsum('ij, k->ijk', c2d0, np.ones(3))
        J = np.einsum('ik, j->ijk', c2d1, np.ones(len(bc)))
        fgdof = space.number_of_global_dofs()
        lgdof = linspace.number_of_global_dofs()
        self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(fgdof, lgdof))

    def setup(self):
        pass

    def solve(self):
        pass
