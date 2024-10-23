
from typing import Sequence

from ..backend import TensorLike
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


class BatchedCosCosCosData():
    def __init__(self, omega: Sequence[float], kappa: float = 1., *,
                 dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        self.kappa = kappa

        if isinstance(omega, TensorLike):
            self.omega = bm.astype(omega, dtype=dtype, copy=True, device=device)
        else:
            self.omega = bm.array(omega, dtype=dtype, device=device)

    @staticmethod
    def domain():
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def solution(self, p: TensorLike): # (*shape, 3)
        pi = bm.pi
        cos = bm.cos
        opx = bm.tensordot(self.omega, p[..., 0], axes=0) * pi
        opy = bm.tensordot(self.omega, p[..., 1], axes=0) * pi
        opz = bm.tensordot(self.omega, p[..., 2], axes=0) * pi
        return cos(opx) * cos(opy) * cos(opz) # (B, *shape)

    dirichlet = solution

    @cartesian
    def gradient(self, p: TensorLike):
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        opx = bm.tensordot(self.omega, p[..., 0], axes=0) * pi
        opy = bm.tensordot(self.omega, p[..., 1], axes=0) * pi
        opz = bm.tensordot(self.omega, p[..., 2], axes=0) * pi
        val = bm.stack([
            -sin(opx) * cos(opy) * cos(opz),
            -cos(opx) * sin(opy) * cos(opz),
            -cos(opx) * cos(opy) * sin(opz)
        ], axis=-1) * pi # (B, *shape, 3)
        return bm.einsum('b, b...d -> b...d', self.omega, val)

    @cartesian
    def source(self, p: TensorLike):
        pi = bm.pi
        cos = bm.cos
        opx = bm.tensordot(self.omega, p[..., 0], axes=0) * pi
        opy = bm.tensordot(self.omega, p[..., 1], axes=0) * pi
        opz = bm.tensordot(self.omega, p[..., 2], axes=0) * pi
        val = 3 * pi**2 * cos(opx) * cos(opy) * cos(opz) # (B, *shape)
        return bm.einsum('b, b... -> b...', self.omega**2, val)

    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike): # (*shape, 3)
        grad = self.gradient(p) # (B, *shape, 3)

        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)

        return bm.einsum('b...d, ...d -> b...', grad, n) # (B, *shape)

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike):
        return self.neumann(p, n) + self.kappa * self.dirichlet(p)
