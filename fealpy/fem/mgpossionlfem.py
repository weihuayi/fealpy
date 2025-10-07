
from fealpy.backend import bm
from fealpy.model import ComputationalModel

from fealpy.mesh import TensorPrismMesh, TriangleMesh

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.fem import KronDirichletBCOperator

from fealpy.sparse import spdiags, coo_matrix, csr_matrix
from fealpy.solver import *

from fealpy.utils import timer
from fealpy.decorator import variantmethod

from scipy.linalg import cholesky, solveh_banded, solve_banded
import scipy.sparse as sp
# from scipy.sparse.linalg import spsolve
from pde import Possion3d
import time

class MGPossionLFEMModel(ComputationalModel):
    """"
    """
    def __init__(self, options=None):
        super().__init__(pbar_log=True, log_level="INFO")

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
        # self.printlevel = options.get('printlevel', 2)
        self.smoother = options.get('smoother', 'LINE')

        self.k = 1     
        self.coarse_time = 0
    
    def set_pde(self, pde:Possion3d):
        self.pde = pde

    def set_mesh(self, tmesh0: TriangleMesh, pmesh: TensorPrismMesh):
        self.tmesh0 = tmesh0
        self.mesh = pmesh
        self.imesh = self.mesh.imesh
        self.tmesh = self.mesh.tmesh
        self.Ny = self.imesh.number_of_nodes()
        self.Nx = self.tmesh.number_of_nodes()
        self.NxNy = self.Nx * self.Ny

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def linear_system(self, mesh: TensorPrismMesh, p):
        """
        """
        self.space0= LagrangeFESpace(mesh.tmesh, p=p)
        self.space1= LagrangeFESpace(mesh.imesh, p=p)

        LDOF0 = self.space0.number_of_local_dofs()
        LDOF1 = self.space1.number_of_local_dofs()
        GDOF0 = self.space0.number_of_global_dofs()
        GDOF1 = self.space1.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF0*LDOF1}, global DOFs: {GDOF0*GDOF1}")
        
        isDDof = self.space0.is_boundary_dof()
        bdIdx = bm.zeros(GDOF0, dtype=bm.float64)
        bdIdx = bm.set_at(bdIdx, isDDof.reshape(-1), 1)
        self.D0x = spdiags(1-bdIdx, 0,GDOF0, GDOF0)
        
        isDDof = self.imesh.boundary_face_flag()
        bdIdx = bm.zeros(GDOF1, dtype=bm.float64)
        bdIdx = bm.set_at(bdIdx, isDDof.reshape(-1), 1)
        self.D0z = spdiags(1-bdIdx, 0, GDOF1, GDOF1)
        
        bform00 = BilinearForm(self.space0)
        bform00.add_integrator(ScalarDiffusionIntegrator())

        bform01 = BilinearForm(self.space0)
        bform01.add_integrator(ScalarMassIntegrator())

        bform10 = BilinearForm(self.space1)
        bform10.add_integrator(ScalarDiffusionIntegrator())

        bform11 = BilinearForm(self.space1)
        bform11.add_integrator(ScalarMassIntegrator())
        
        self.space = LagrangeFESpace(mesh, p=p)
        self.uh = self.space.function()

        A00 = bform00.assembly()
        A01 = bform01.assembly()
        A10 = bform10.assembly()
        A11 = bform11.assembly()
        # import ipdb;ipdb.set_trace()
        self.A00 = self.D0x @ A00 @ self.D0x
        self.A01 = self.D0x @ A01 @ self.D0x
        self.A10 = self.D0z @ A10 @ self.D0z
        self.A11 = self.D0z @ A11 @ self.D0z
        
        A = (sp.kron(A00.to_scipy(), A11.to_scipy()) + sp.kron(A01.to_scipy(), A10.to_scipy())).tocoo()
        
        from fealpy.sparse import COOTensor
        A = COOTensor(
            indices=bm.stack([A.row, A.col], axis=0),
            values=A.data,
            spshape=A.shape
        )
        
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source))
        F = lform.assembly()

        return A, F

    def setup(self, A):
        """计算限制算子、插值算子, 以及每层 A 的 LL^T 分解
        """
        # 1. 建立初步的算子存储结构
        self.A = [A.to_scipy()]
        self.B = [ ] # A 的近似矩阵
        self.P = [ ] # 延拓算子
        self.R = [ ] # 限制矩阵
        self.L = [ ] # 下三角 
        self.U = [ ] # 上三角
        self.ab = []

        smoother = self.smoother
        mesh = self.tmesh0

        # 2. 计算 P 和 R
        IM = mesh.uniform_refine(n=self.level-1, returnim=True)
        nshape = IM[0].shape[0]
        op = spdiags(bm.ones((nshape,)), 0, nshape, nshape)
        Iz = csr_matrix(bm.eye(self.Ny)).to_scipy()

        for i in range(self.level - 1):
            p =  (sp.kron(IM[i].to_scipy(), Iz)).tocoo()
            r = p.T

            self.P.append(p)
            self.R.append(r)

            k1 = op.T @ self.A00 @ op
            k2 = op.T @ self.A01 @ op
            k3 = op.T @ op
            k4 = op.T @ self.D0x @ op

            B = (
                sp.kron(sp.diags(k1.to_scipy().diagonal()), self.A11.to_scipy()) +
                sp.kron(sp.diags(k2.to_scipy().diagonal()), self.A10.to_scipy()) +
                sp.kron(sp.diags(k3.to_scipy().diagonal()), Iz) -
                sp.kron(sp.diags(k4.to_scipy().diagonal()), self.D0z.to_scipy())
            )

            if i < (self.level - 1):
                op = op @ IM[i]
            
            k1 = op.T @ self.A00 @ op
            k2 = op.T @ self.A01 @ op
            k3 = op.T @ op
            k4 = op.T @ self.D0x @ op

            self.A.append(
                sp.kron(k1.to_scipy(), self.A11.to_scipy()) +
                sp.kron(k2.to_scipy(), self.A10.to_scipy()) +
                sp.kron(k3.to_scipy(), Iz) -
                sp.kron(k4.to_scipy(), self.D0z.to_scipy())
            )

            self.B.append(B)
                
    def vcycle(self, r, J=None):
        """
        solve equations Ae = r in each level  
        """ 
          
        if J is None:
            J = self.level
        
        ri = [None] * J
        ei = [None] * J
        ri[0] = r

        # 粗化
        start = time.time()
        for i in range(J-1):
            ei[i] = self.linesmoother(ri[i], i)

            for _ in range(self.mu):
                ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)

            ri[i+1] = self.R[i] @ (ri[i] - self.A[i] @ ei[i])
        print(time.time()-start, '粗化时间 time \n')

        # 粗网格求解
        if self.coarsegridsolver == 'direct':
            start = time.time()
            ei[-1] = cg(self.A[-1], ri[-1], maxit=1000, atol=1e-9, rtol=1e-9)
            print(time.time()-start, ' 最粗网格第k次求解所花时间\n')
            self.coarse_time += time.time()-start
            
        else:
            pass
        
        # 插值
        start = time.time()
        for i in range(J-2, -1, -1):
            ei[i] += self.P[i] @ ei[i+1]
            ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)

            for _ in range(self.mu):
                ei[i] += self.linesmoother(ri[i] - self.A[i] @ ei[i], i)
        print(time.time()-start, '插值时间 time \n')
        return ei[0]
    
    def linesmoother(self, r, J):
        """
        求解 LU e = r
        """
        # L = self.L
        # U = self.U
        # z = spsolve(csr_matrix(L[J]), r, solver='scipy')
        # e = spsolve(csr_matrix(U[J]), z, solver='scipy')
        # e = solve_banded(self.ab[J], r, lower=False)
        # import pyamg
        # ml = pyamg.ruge_stuben_solver(self.B[J])
        # e = ml.solve(r, maxiter=5, cycle='V')
        # import ipdb;ipdb.set_trace() 
        hb = bm.zeros((2, self.B[J].shape[0]))
        hb[0, 1:] = self.B[J].diagonal(1)
        hb[1, :] = self.B[J].diagonal(0)
        e = solveh_banded(hb, r, lower=False)
        # e = gauss_seidel(self.B[J], r, maxit=100, atol=1e-6, rtol=1e-6)
        # e, info = cg(self.B[J], r, maxit=10, atol=1e-6, rtol=1e-6, returninfo=True)
        # print(info)
        # e = spsolve(self.B[J], r)
        e = 0.75 * e

        return e
    
    @variantmethod('cg')
    def solve(self, A, F):
        # self.uh[:] = spsolve(A, F, solver='mumps')
        # import scipy.sparse as sp
        # from scipy.sparse.linalg import cg
        # import pyamg
        # # import ipdb;ipdb.set_trace()
        # ml = pyamg.smoothed_aggregation_solver(A.to_scipy())
        # M = ml.aspreconditioner()
        
        # # CG 求解
        # self.uh[:], info = cg(A.to_scipy(), F, rtol=1e-8, atol=1e-8, maxiter=100, M=M)
        self.uh[:], info = cg(A, F, maxit=1000, atol=1e-9, rtol=1e-9, returninfo=True)
        # print(info)
        return self.uh
    
    @solve.register('gmg')
    def solve(self, A, F):
        # initial set up
        start = time.time()
        self.setup(A)
        print(time.time()-start, '初始时间 time \n')

        k = 0
        x = bm.zeros_like(F) if self.x0 == None else self.x0
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
        cost_time = time.time() - start
        
        print(f"dof: {self.NxNy:2.0f},  "
            f"level: {self.level:2.0f},  "
            f"smoothing: {self.mu:2.0f},  "
            f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"coarse grid: {self.A[-1].shape[0]:2.0f},  "
            f"time = {cost_time:2.2g} s")

        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return x

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        tmr = timer()
        next(tmr)
        A, F = self.linear_system(self.mesh, self.p)
        tmr.send(f'初步组装线性系统时间')
        A, F = DirichletBC(self.space, 
                           gd=self.pde.dirichlet, 
                           threshold=self.pde.is_dirichlet_boundary).apply(A, F)
        tmr.send(f'边界处理时间')
        self.uh[:] = self.solve['gmg'](A, F)
        tmr.send(f'求解器时间')
        next(tmr)
        print(self.coarse_time)
        l2 = self.postprocess()
        self.logger.info(f"L2 Error: {l2}.")

    def postprocess(self):
        l2 = self.mesh.error(self.pde.solution, self.uh)
        return l2

