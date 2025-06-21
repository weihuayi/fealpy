
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg,  dsolve,  gmres, lgmres, LinearOperator, spsolve_triangular
from scipy.sparse.linalg import spilu
import pyamg
from timeit import default_timer as dtimer 

from ..decorator import timer

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))

class LinearElasticityRLFEMFastSolver():
    def __init__(self, lam, mu, M, G, isBdDof, P=None):
        """

        Notes
        -----
        M: 质量矩阵 (gdof, gdof)
        G: 恢复矩阵 (X, Y, Z)
        P: 预条件矩阵
        isBdDof: Dirichlet 边界自由度标记 (gdof, )
        """

        self.GD = len(G) 
        self.gdof = M.shape[0]

        self.isBdDof = isBdDof
        self.lam = lam
        self.mu = mu

        self.M = M
        self.G = G

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
        r = b.copy()
        r = r.reshape(GD, -1)
        val = r[:, isBdDof]
        r[:, isBdDof] = 0.0

        t = [GD*[None], GD*[None]]
        if GD == 3:
            t += [GD*[None]]

        for i in range(GD):
            for j in range(GD):
                t[i][j] = M@(G[j]@r[i, :])

        r[:] = 0.0
        r[0, :] += (2*mu + lam)*(t[0][0]@G[0]) # X^T*M*X
        r[0, :] += mu*(t[0][1]@G[1])
        r[0, :] += mu*(t[0][2]@G[2])

        r[0, :] += lam*(t[1][1]@G[0]) 
        r[0, :] += mu*(t[1][0]@G[1])

        r[0, :] += lam*(t[2][2]@G[0])
        r[0, :] += mu*(t[2][0]@G[2])


        r[1, :] += lam*(t[0][0]@G[1]) 
        r[1, :] += mu*(t[0][1]@G[0])

        r[1, :] += (2*mu + lam)*(t[1][1]@G[1]) # X^T*M*X
        r[1, :] += mu*(t[1][2]@G[2])
        r[1, :] += mu*(t[1][0]@G[0])

        r[1, :] += lam*(t[2][2]@G[1])
        r[1, :] += mu*(t[2][1]@G[2])


        r[2, :] += lam*(t[0][0]@G[2]) 
        r[2, :] += mu*(t[0][2]@G[0])

        r[2, :] += lam*(t[1][1]@G[2])
        r[2, :] += mu*(t[1][2]@G[1])

        r[2, :] += (2*mu + lam)*(t[2][2]@G[2]) # X^T*M*X
        r[2, :] += mu*(t[2][0]@G[0])
        r[2, :] += mu*(t[2][1]@G[1])

        r[:, isBdDof] = val
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
        isBdDof = self.isBdDof
        M = self.M
        G = self.G

        # 处理 Dirichlet 右端边界条件
        r = self.linear_operator(uh.T.flat)
        F.T.flat -= r
        F[isBdDof] = uh[isBdDof]

        A = LinearOperator((GD*gdof, GD*gdof), matvec=self.linear_operator)
                
        counter = IterationCounter()
        uh.T.flat, info = cg(A, F.T.flat, tol=tol, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)

        return uh 

    def cg(self, A, F, uh):
        counter = IterationCounter()
        uh.T.flat, info = cg(A, F.T.flat, tol=1e-8, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)
        return uh 
