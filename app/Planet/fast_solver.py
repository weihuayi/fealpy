
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

    @timer
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

    @timer
    def linear_operator_2(self, b):
        b = self.D@b
        return b

    @timer
    def solve(self, uh, F):
        rdof = self.rdof
        gdof = self.gdof

        A = LinearOperator((rdof, rdof), matvec=self.linear_operator_1)
        a = F[:rdof]

        if self.ctx.myid == 0:
            self.ctx.set_rhs(b)
        self.ctx.run(job=3)

        a -= self.B@F[rdof:]

        uh[:rdof].T.flat, info = cg(A, a, tol=1e-8)

        P = LinearOperator((gdof, gdof), matvec=self.linear_operator_2)
        uh[rdof:].T.flat, info = cg(P, F[rdof:]-uh[:rdof]@self.B, tol=1e-8)
