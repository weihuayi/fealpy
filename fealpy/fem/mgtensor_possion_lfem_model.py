from typing import Optional, Union
from fealpy.backend import bm

from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.model.mgtensor_possion import MGTensorPossionPDEDataT

from fealpy.sparse import spdiags, coo_matrix, csr_matrix
from fealpy.solver import cg

from fealpy.utils import timer
from fealpy.decorator import variantmethod


class SumOperator:
    def __init__(self, *ops):
        self.ops = ops
        self.shape = ops[0].shape

    def __matmul__(self, x):
        y = 0
        for op in self.ops:
            y = y + (op @ x)
        return y


class LinearOperator:
    def __matmul__(self, x):
        raise NotImplementedError
    
    def __add__(self, other):
        return SumOperator(self, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)


class KronOperator(LinearOperator):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.m0, self.n0 = A.shape
        self.m1, self.n1 = B.shape
        self.shape = (self.m0*self.m1, self.n0*self.n1)

    def __matmul__(self, x):
        X = bm.reshape(x, (self.n0, self.m1))
        X = csr_matrix(X)
        Y = self.A @ X @ self.B
        Y = Y.to_dense().ravel()
        return Y


class MGTensorPossionLFEMModel(ComputationalModel):
    """"Multigrid solver for Poisson equations defined on 
            tensor-product grids using the Linear Finite Element Method (LFEM).

    """
    def __init__(self, options=None):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEModelManager("mgtensor_possion")

        if options is None:
            options = {} 
            
        self.level = options.get('level')

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.mu = options.get('smoothingstep', 1)
        self.solver = options.get('solver', 'VCYCLE')

        self.preconditioner = options.get('preconditioner', 'none')
        self.coarsegridsolver = options.get('coarsegridsolver', 'direct')
        self.smoother = options.get('smoother', 'LINE')
    
    def set_pde(self, pde: Union[MGTensorPossionPDEDataT, str, int]):
        """
        """
        if isinstance(pde, int):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, tmesh: TriangleMesh, imesh: IntervalMesh):
        self.tmesh = tmesh
        self.IM = tmesh.uniform_refine(n=self.level-1, returnim=True)
        self.imesh = imesh
        self.Ny = self.imesh.number_of_nodes()
        self.Nx = self.tmesh.number_of_nodes()
        self.NxNy = self.Nx * self.Ny
        self.x0 = bm.zeros((self.NxNy,), dtype=bm.float64)
        
        tnode = tmesh.entity('node')
        inode = imesh.entity('node')
        
        self.node = bm.concat([bm.repeat(tnode, inode.shape[0], axis=0), 
                          bm.tile(inode.T, tnode.shape[0]).T], axis=1)

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def linear_system(self):
        """
        """
        p = self.p
        self.space0= LagrangeFESpace(self.tmesh, p=p)
        self.space1= LagrangeFESpace(self.imesh, p=p)

        LDOF0 = self.space0.number_of_local_dofs()
        LDOF1 = self.space1.number_of_local_dofs()
        GDOF0 = self.space0.number_of_global_dofs()
        GDOF1 = self.space1.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF0*LDOF1}, global DOFs: {GDOF0*GDOF1}")
        
        isDDof0 = self.space0.is_boundary_dof()
        bdIdx0 = bm.zeros(GDOF0, dtype=bm.float64)
        bdIdx0 = bm.set_at(bdIdx0, isDDof0.reshape(-1), 1)
        self.D0x = spdiags(1-bdIdx0, 0,GDOF0, GDOF0)
        
        isDDof1 = self.imesh.boundary_face_flag()
        bdIdx1 = bm.zeros(GDOF1, dtype=bm.float64)
        bdIdx1 = bm.set_at(bdIdx1, isDDof1.reshape(-1), 1)
        self.D0z = spdiags(1-bdIdx1, 0, GDOF1, GDOF1)
        
        bform00 = BilinearForm(self.space0)
        bform00.add_integrator(ScalarDiffusionIntegrator())

        bform01 = BilinearForm(self.space0)
        bform01.add_integrator(ScalarMassIntegrator())

        bform10 = BilinearForm(self.space1)
        bform10.add_integrator(ScalarDiffusionIntegrator())

        bform11 = BilinearForm(self.space1)
        bform11.add_integrator(ScalarMassIntegrator())

        A00 = bform00.assembly()
        A01 = bform01.assembly()
        A10 = bform10.assembly()
        A11 = bform11.assembly()

        self.A00 = self.D0x @ A00 @ self.D0x
        self.A01 = self.D0x @ A01 @ self.D0x
        self.A10 = self.D0z @ A10 @ self.D0z
        self.A11 = self.D0z @ A11 @ self.D0z
        
        Ix = csr_matrix(bm.eye(self.Nx))
        Iz = csr_matrix(bm.eye(self.Ny))
        # import scipy.sparse as sp
        # A = (
        #     sp.kron(self.A00.to_scipy(), self.A11.to_scipy()) + 
        #     sp.kron(self.A01.to_scipy(), self.A10.to_scipy()) + 
        #     sp.kron(Ix.to_scipy(), Iz.to_scipy()) + 
        #     sp.kron(self.D0x.to_scipy(), -self.D0z.to_scipy())
        # )
        A = (
            KronOperator(self.A00, self.A11) + 
            KronOperator(self.A01, self.A10) + 
            KronOperator(Ix, Iz) + 
            KronOperator(self.D0x, -self.D0z)
        )

        index_dof = bm.arange(self.NxNy)[~((~isDDof0[:, None]) * (~isDDof1[None, :])).ravel()]
        gd = self.pde.dirichlet
        threshold = self.pde.is_dirichlet_boundary
        uh = self.x0
        ipoint = self.node[index_dof]
        flag = threshold(ipoint)
        
        index_dof = index_dof[flag]
        isBdDof = bm.zeros(self.NxNy, dtype=bm.bool)
        isBdDof = bm.set_at(isBdDof, index_dof, True)

        gd = gd(self.node[isBdDof])
        uh = bm.set_at(uh, (..., isBdDof), gd)
        self.uh = bm.zeros((self.NxNy,), dtype=bm.float64)
        
        F = bm.zeros((self.NxNy,), dtype=bm.float64)
        # Todo: X 是否可以用稀疏矩阵储存？
        X = csr_matrix(bm.reshape(uh, (self.Nx, self.Ny)))
        F = F - (A00 @ X @ A11 + A01 @ X @ A10).to_dense().ravel()
        F = bm.set_at(F, isBdDof, uh[isBdDof])
        return A, F

    def setup(self, A):
        """Compute restriction and interpolation operators.
        """
        self.A = [A]
        self.B = [ ] 
        self.P = [ ] 
        self.R = [ ] 
        self.L = [ ] 
        self.U = [ ] 
        self.ab = []

        # Compute P and R.
        IM = self.IM
        nshape = IM[0].shape[0]
        op = spdiags(bm.ones((nshape,)), 0, nshape, nshape)
        Iz = csr_matrix(bm.eye(self.Ny))

        for i in range(self.level - 1):
            p = KronOperator(IM[i], Iz)
            r = KronOperator(IM[i].T, Iz)

            self.P.append(p)
            self.R.append(r)

            k1 = op.T @ self.A00 @ op
            k2 = op.T @ self.A01 @ op
            k3 = op.T @ op
            k4 = op.T @ self.D0x @ op

            B = (
                KronOperator(k1.diags(), self.A11) +
                KronOperator(k2.diags(), self.A10) +
                KronOperator(k3.diags(), Iz) +
                KronOperator(k4.diags(), -self.D0z)
            )

            if i < (self.level - 1):
                op = op @ IM[i]
            
            k1 = op.T @ self.A00 @ op
            k2 = op.T @ self.A01 @ op
            k3 = op.T @ op
            k4 = op.T @ self.D0x @ op

            self.A.append(
                KronOperator(k1, self.A11) +
                KronOperator(k2, self.A10) +
                KronOperator(k3, Iz) +
                KronOperator(k4, -self.D0z)
            )

            self.B.append(B)
            
    def vcycle(self, r, J=None):
        """solve equations Ae = r in each level  
        """ 
          
        if J is None:
            J = self.level
        
        ri = [None] * J
        ei = [None] * J
        ri[0] = r

        # 粗化
        for i in range(J-1):
            ei[i] = self.linesmoother(ri[i], i)

            for _ in range(self.mu):
                ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)

            ri[i+1] = self.R[i] @ (ri[i] - self.A[i] @ ei[i])

        # 粗网格求解
        if self.coarsegridsolver == 'direct':
            ei[-1] = cg(self.A[-1], ri[-1], maxit=1000, atol=1e-9, rtol=1e-9)
            
        else:
            pass
        
        # 插值
        for i in range(J-2, -1, -1):
            ei[i] += self.P[i] @ ei[i+1]
            ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)

            for _ in range(self.mu):
                ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)
        return ei[0]
    
    def linesmoother(self, r, J):
        """Solve LUe = r.
        """
        # L = self.L
        # U = self.U
        # z = spsolve(csr_matrix(L[J]), r, solver='scipy')
        # e = spsolve(csr_matrix(U[J]), z, solver='scipy')
        # e = solve_banded(self.ab[J], r, lower=False)
        e = cg(self.B[J], r, maxit=100, atol=1e-6, rtol=1e-6)
        # e = spsolve(self.B[J], r)
        e = 0.75 * e

        return e
    
    @variantmethod('cg')
    def solve(self, A, F):
        self.uh[:], info = cg(A, F, maxit=1000, atol=1e-9, rtol=1e-9, returninfo=True)
        print(info)
        return self.uh
    
    @solve.register('gmg')
    def solve(self, A, F):
        # initial set up
        self.setup(A)

        k = 0
        x = self.x0
        r = F - A @ x
        nb = bm.linalg.norm(F)
        err = bm.zeros((self.maxIt, 2), dtype=bm.float64)

        if nb > bm.finfo(float).eps:
            err[0, :] = bm.linalg.norm(r) / nb
        else:
            err[0, :] = bm.linalg.norm(r)

        if self.solver == 'VCYCLE':
            print('Multigrid Vcycle Iteration \n')
            while (bm.max(err[k, :]) > self.tol) & (k <= self.maxIt):
                k = k + 1
                Br = self.vcycle(r)
                x = x + Br
                r = r - A @ Br
                err[k, 0] = bm.sqrt(bm.abs(Br.T @ r / (x.T @ F)))
                err[k, 1] = bm.linalg.norm(r) / nb

                print(
                    f'MG Vcycle iter: {k:2d},   '
                    f'err = {bm.max(err[k, :]):8.4e}'
                )
            err = err[:k, :]
            itStep = k

        elif self.solver == 'WCYCLE':
            pass
        
        # Output
        print(f"dof: {self.NxNy:2.0f},  "
            f"level: {self.level:2.0f},  "
            f"smoothing: {self.mu:2.0f},  "
            f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"coarse grid: {self.A[-1].shape[0]:2.0f},  ")

        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return x

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        tmr = timer()
        next(tmr)
        A, F = self.linear_system()
        tmr.send(f'初步组装线性系统时间')
        self.uh = self.solve['gmg'](A, F)
        tmr.send(f'求解器时间')
        next(tmr)
        err = self.postprocess()
        self.logger.info(f"L2 point Error: {err}.")

    def postprocess(self):
        err = bm.sqrt(bm.mean((self.pde.solution(self.node) - self.uh)**2))
        return err
    


