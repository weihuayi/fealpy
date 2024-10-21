import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg, spsolve, LinearOperator
from timeit import default_timer as timer
import pyamg
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..femmodel.doperator import stiff_matrix


class HOFEMFastSovler():
    def __init__(self, A, space, integrator, measure):
        self.A = A


        self.DL = tril(A).tocsr()
        self.U = triu(A, k=1).tocsr()

        self.DU = triu(A).tocsr()
        self.L =  tril(A, k=-1).tocsr()

        linspace = LagrangeFiniteElementSpace(space.mesh, 1)

        # construct amg solver for linear 
        A1 = stiff_matrix(linspace, integrator, measure)
        isBdDof = linspace.boundary_dof()
        bdIdx = np.zeros((A1.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A1.shape[0], A1.shape[0])
        T = spdiags(1-bdIdx, 0, A1.shape[0], A1.shape[0])
        A1 = T@A1@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(A1)  

        # Get interpolation matrix 
        NC = space.mesh.number_of_cells()
        bc = space.dof.multiIndex/space.p
        val = np.tile(bc, (NC, 1))
        c2d0 = space.cell_to_dof()
        c2d1 = linspace.cell_to_dof()

        I = np.einsum('ij, k->ijk', c2d0, np.ones(3))
        J = np.einsum('ik, j->ijk', c2d1, np.ones(len(bc)))
        gdof = space.number_of_global_dofs()
        lgdof = linspace.number_of_global_dofs()
        self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(gdof, lgdof))

    def solve(self, b, tol=1e-13):
        gdof = self.DL.shape[0]
        P = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        start = timer()
        x, info = cg(self.A, b, M=P, tol=tol)
        end = timer()
        print("Solve time:", end-start, " with convergence info: ", info)
        return x

    def linear_operator(self, r):
        gdof = self.DL.shape[0]
        u = np.zeros(gdof, dtype=np.float)
        for i in range(6):
            u[:] = spsolve(self.DL, r - self.U@u, permc_spec="NATURAL") 

        r0 = r - (self.DL@u + self.U@u)
        u0 = self.ml.solve(self.PI.transpose()@r0, tol=1e-13, accel='cg')

        u += self.PI@u0
        for i in range(6):
            u[:] = spsolve(self.DU, r - self.L@u, permc_spec="NATURAL") 

        return u
