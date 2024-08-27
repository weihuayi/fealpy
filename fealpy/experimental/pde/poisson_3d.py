from ..backend import backend_manager as bm
from ..decorator import cartesian

class CosCosCosData:

    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.cos(bm.pi*x)*bm.cos(bm.pi*y)*bm.cos(bm.pi*z)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = bm.stack([
            -pi*sin(pi*x)*cos(pi*y)*cos(pi*z),
            -pi*cos(pi*x)*sin(pi*y)*cos(pi*z),
            -pi*cos(pi*x)*cos(pi*y)*sin(pi*z)
        ], axis=-1)
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*bm.pi**2*bm.cos(bm.pi*x)*bm.cos(bm.pi*y)*bm.cos(bm.pi*z)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NF, NQ, 3)
        n: (NF, 3)

        grad*n : (NQ, NE, 3)
        """
        grad = self.gradient(p) # (NF, NQ, 3)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        val = bm.einsum('fqd, fqd -> fq', grad, n) # (NF, NQ)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 3)
        val = bm.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = bm.array([1.0], dtype=bm.float64).reshape(shape)
        val = bm.sum(grad*n, axis=-1) + self.solution(p) 
        return val, kappa

