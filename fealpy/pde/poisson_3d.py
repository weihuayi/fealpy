
from typing import List, Tuple, Union

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..decorator import cartesian

class CosCosCosData:
    def __init__(self, omega: int = 1, kappa: float = 1.):
        self.omega = omega
        self.kappa = kappa

    @staticmethod
    def domain():
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def solution(self, p: TensorLike):
        """ the exact solution
        """
        op = bm.pi * self.omega
        cos = bm.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = cos(op*x) * cos(op*y) * cos(op*z)
        return u

    @cartesian
    def gradient(self, p: TensorLike):
        """ The gradient of the exact solution
        """
        op = self.omega * bm.pi
        sin = bm.sin
        cos = bm.cos
        opx = p[..., 0] * op
        opy = p[..., 1] * op
        opz = p[..., 2] * op
        val = bm.stack([
            -op * sin(opx) * cos(opy) * cos(opz),
            -op * cos(opx) * sin(opy) * cos(opz),
            -op * cos(opx) * cos(opy) * sin(opz)
        ], axis=-1)
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p: TensorLike):
        op = bm.pi * self.omega
        cos = bm.cos
        opx = p[..., 0] * op
        opy = p[..., 1] * op
        opz = p[..., 2] * op
        val = 3 * op**2 * cos(opx) * cos(opy) * cos(opz)
        return val

    # @cartesian
    # def dirichlet(self, p):
    #     """Dilichlet boundary condition
    #     """
    #     return self.solution(p)
    dirichlet = solution

    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike):
        grad = self.gradient(p) # (*shape, 3)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        val = bm.einsum('...d, ...d -> ...', grad, n) # (*shape)
        return val

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike):
        # grad = self.gradient(p) # (*shape, 3)
        # val = bm.sum(grad*n, axis=-1)
        # shape = len(val.shape)*(1, )
        # kappa = bm.array([1.0], dtype=bm.float64).reshape(shape)
        # val = bm.sum(grad*n, axis=-1) + self.solution(p) 
        # return val, kappa
        return self.neumann(p, n) + self.kappa * self.dirichlet(p)


class BatchedCosCosCosData():
    def __init__(self, omega: Union[List[int], Tuple[int, ...]], kappa: float = 1., *,
                 dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        self.kappa = kappa
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
