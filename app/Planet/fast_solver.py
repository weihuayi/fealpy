
from fealpy.decorator import timer

class PlanetFastSovler():
    def __init__(self, D, B, ctx):

        self.B = B
        if ctx.myid == 0:
            ctx.set_centralized_sparse(D)

        ctx.run(job=4) # Analysis + Factorrization
        self.ctx = ctx

    def set_matrix(self, Ak):
        self.k = AAk

    def linear_operator(self, b):
        '''
        
        (A - B D^{-1} C) b

        '''
        r = self.Ak@b

        b = b@self.B

        if self.ctx.myid == 0:
            self.ctx.set_rhs(b)
        self.ctx.run(job=3)

        r -= self.B@b

        return r

    def solve(self, uh, F):

        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        uh.T.flat, info = cg(A, F, tol=1e-8)
        pass


    
