
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg, inv, dsolve,  gmres, lgmres, LinearOperator, spsolve_triangular
import pyamg

from ..decorator import timer

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class LinearElasticityRLFEMFastSolver():
    def __init__(self, mu, lam, M, G, P, isBdDof):
        """

        Notes
        -----
        M: 质量矩阵 (gdof, gdof)
        G: 恢复矩阵 [X, Y, Z] 
        P: 预条件矩阵
        isBdDof: Dirichlet 边界自由度标记 (gdof, )
        """

        self.GD = len(G) 
        self.gdof = P.shape[0]

        self.isBdDof = isBdDof
        self.lam = lam
        self.mu = mu

        self.M = M
        self.G = G

        # 处理预条件子的边界条件
        bdIdx = np.zeros(P.shape[0], dtype=np.int_)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, P.shape[0], P.shape[0])
        T = spdiags(1-bdIdx, 0, P.shape[0], P.shape[0])
        P = T@P@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(P) 

    def linear_operator(self, b):
        """

        Notes
        -----
        b: (GD*gdof, )
        """
        lam = self.lam
        mu = self.mu
        GD = self.GD

        M = self.M
        G = self.G

        # A@b
        isBdDof = self.isBdDof
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        val = b[:, isBdDof]
        b[:, isBdDof] = 0.0

        t = GD*[GD*[None]]
        for i in range(GD):
            for j in range(GD):
                t[i][j] = M@(G[j]@b[i, :])

        for i in range(GD):
            for j in range(GD):
                if i == j:
                    r[i, :] += (2*mu + lam)*t[i][i]@G[i]
                else:
                    r[i, :] += lam*t[j][j]@G[i]
                    r[i, :] += mu*t[i][j]@G[j]
                    r[i, :] += mu*t[j][i]@G[j]
        r[:, isBdDof] = val
        return r.reshape(-1)

    def preconditioner(self, b):
        GD = self.GD
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        for i in range(GD):
            r[i] = self.ml.solve(b[i], tol=1e-8, accel='cg')       
        return r.reshape(-1)

    @timer
    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----

        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        lam = self.lam
        mu = self.mu
        GD = self.GD
        gdof = self.gdof
        M = self.M
        G = self.G

        # 处理 Dirichlet 右端边界条件
        isBdDof = self.isBdDof
        t = GD*[GD*[None]]
        for i in range(GD):
            for j in range(GD):
                t[i][j] = M@(G[j]@uh[:, i])

        for i in range(GD):
            for j in range(GD):
                if i == j:
                    F[:, i] -= (2*mu + lam)*t[i][i]@G[i]
                else:
                    F[:, i] -= lam*t[j][j]@G[i]
                    F[:, i] -= mu*t[i][j]@G[j]
                    F[:, i] -= mu*t[j][i]@G[j]
        F[isBdDof] = uh[isBdDof]

        A = LinearOperator((GD*gdof, GD*gdof), matvec=self.linear_operator)
        P = LinearOperator((GD*gdof, GD*gdof), matvec=self.preconditioner)
                
        counter = IterationCounter()
        uh.T.flat, info = cg(A, F.T.flat, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)

        return uh 

    def cg(self, A, F, uh):
        counter = IterationCounter()
        uh.T.flat, info = cg(A, F.T.flat, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)
        return uh 

class LinearElasticityLFEMFastSolver():
    def __init__(self, A, P, isBdDof):
        """
        Notes
        -----
        A: [[A00, A01], [A10, A11]] (2*gdof, 2*gdof)
           [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]] (3*gdof, 3*gdof)
        P: 预条件子 (gdof, gdof)

        这里的边界条件处理放到矩阵和向量的乘积运算当中, 所心不需要修改矩阵本身
        """
        self.GD = len(A) 
        self.gdof = P.shape[0]

        self.A = A
        self.isBdDof = isBdDof

        # 处理预条件子的边界条件
        bdIdx = np.zeros(P.shape[0], dtype=np.int_)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, P.shape[0], P.shape[0])
        T = spdiags(1-bdIdx, 0, P.shape[0], P.shape[0])
        P = T@P@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(P) 

    def linear_operator(self, b):
        """
        Notes
        -----
        b: (2*gdof, )
        """
        GD = self.GD
        isBdDof = self.isBdDof
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        val = b[:, isBdDof]
        b[:, isBdDof] = 0.0
        for i in range(GD):
            for j in range(GD):
                r[i] += self.A[i][j]@b[j]
        r[:, isBdDof] = val
        return r.reshape(-1)

    def preconditioner(self, b):
        GD = self.GD
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        for i in range(GD):
            r[i] = self.ml.solve(b[i], tol=1e-8, accel='cg')       
        return r.reshape(-1)

    @timer
    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----

        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        GD = self.GD
        gdof = self.gdof

        # 处理 Dirichlet 右端边界条件
        for i in range(GD):
            for j in range(GD):
                F[:, i] -= self.A[i][j]@uh[:, j]
        F[isBdDof] = uh[isBdDof]

        A = LinearOperator((GD*gdof, GD*gdof), matvec=self.linear_operator)
        P = LinearOperator((GD*gdof, GD*gdof), matvec=self.preconditioner)
                
        uh.T.flat, info = pcg(A, F.T.flat, M=P, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)

        return uh 

class LagrangeFEMFastSolver():
    def __init__(self, A, F, P):
        """


        Notes
        -----
            求解高次拉格朗日有限元的快速算法
        """
        self.A = A
        self.F = F
        self.P = P

class SaddlePointFastSolver():
    def __init__(self, A, F):
        """

        Notes
        -----
            A = (M, B, C), C 可以是 None
            F = (F0, F1), 

            求解如下离散代数系统 
            M   x0 + B x1 = F0 
            B^T x0 + C x1 = F1

        TODO:

        """
        self.A = A
        self.F = F

        M = A[0]
        B = A[1]
        C = A[2]

        self.D = 1.0/M.diagonal() # M 矩阵的对角线的逆
        # S 相当于间断元的刚度矩阵
        S = (B.T@spdiags(self.D, 0, M.shape[0], M.shape[1])@B).tocsr()
        self.ml = pyamg.ruge_stuben_solver(S) # 这里要求必须有网格内部节点 

        # TODO：把间断元插值到连续元线性元空间，然后再做 AMG

    def linear_operator(self, b):
        M = self.A[0]
        B = self.A[1]
        m = M.shape[0]
        n = B.shape[1]
        r = np.zeros_like(b)
        r[:m] = M@b[:m] + B@b[m:]
        r[m:] = B.T@b[:m]
        return r

    def diag_preconditioner(self, b):
        D = self.D
        m = self.A[0].shape[0]
        n = self.A[1].shape[1]

        r = np.zeros_like(b)

        b0 = b[:m]
        b1 = b[m:]

        r[:m] = b0*D
        r[m:] = self.ml.solve(b1, tol=1e-8, accel='cg')       
        return r 
    
    @timer
    def solve(self, tol=1e-8):
        M = self.A[0]
        B = self.A[1]
        C = self.A[2]

        m = M.shape[0]
        n = B.shape[1]
        gdof = m + n

        counter = IterationCounter()
        F = np.r_[self.F[0], self.F[1]]
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.diag_preconditioner)
        x, info = lgmres(A, F, M=P, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)

        return x[:m], x[m:] 



