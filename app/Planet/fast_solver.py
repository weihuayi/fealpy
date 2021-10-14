
import pyamg
import numpy as np

from fealpy.decorator import timer
from scipy.sparse.linalg import cg, LinearOperator

class PlanetFastSovler():
    def __init__(self, D, B, ctx):
        rdof = B.shape[0]
        gdof = B.shape[1]

        self.rdof = rdof
        self.gdof = gdof

        self.B = B
        self.D = D
        if ctx.myid == 0:
            ctx.set_centralized_sparse(D)

        ctx.run(job=4) # Analysis + Factorization
        self.ctx = ctx

    def set_matrix(self, Ak):
        self.Ak = Ak

    def linear_operator_1(self, b):
        """ 
        (A - B D^{-1} C) b
        """
        r = self.Ak@b

        b = b@self.B

        if self.ctx.myid == 0:
            self.ctx.set_rhs(b)
        self.ctx.run(job=3)

        r -= self.B@b

        return r

    def linear_operator_2(self, b):
        b = self.D@b
        return b

    @timer
    def solve_1(self, u, F):
        rdof = self.rdof
        gdof = self.gdof

        A = LinearOperator((rdof, rdof), matvec=self.linear_operator_1)
        a = F[:rdof]

        b = np.zeros(gdof, dtype=np.float64)
        b[:] = F[rdof:]

        if self.ctx.myid == 0:
            self.ctx.set_rhs(b)
        self.ctx.run(job=3)

        a -= self.B@b

        u[:rdof].T.flat, info = cg(A, a, tol=1e-8)

        P = LinearOperator((gdof, gdof), matvec=self.linear_operator_2)
        u[rdof:].T.flat, info = cg(P, F[rdof:]-u[:rdof]@self.B, tol=1e-8)
        return u
    
    def linear_operator_3(self, b):
        """ 
        Ax + By = a
        Cx + Dy = b
        """
        rdof = self.rdof

        x = b[:rdof]
        y = b[rdof:]

        r = np.zeros_like(b)

        r[:rdof] = self.Ak@x + self.B@y
        r[rdof:] = x@self.B + self.D@y

        return r

    @timer
    def solve_2(self, u, F):
        dof = F.shape[0]

        A = LinearOperator((dof, dof), matvec=self.linear_operator_3)

        u.T.flat, info = cg(A, F, tol=1e-8)
        return u
