from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..box_domain_mesher import BoxDomainMesher2d
from ...decorator import cartesian

class EXP0003(BoxDomainMesher2d):
    """
    2D Helmholtz problem with complex Robin boundary condition:

        -Δu(x, y) - k^2·u(x, y) = 0,       in Ω = (0, 1) x (0, 1)
         ∂u/∂n + i·k·u = g(x, y),          on ∂Ω

    where u(x, y) = exp(i(k1 x + k2 y)) is the exact solution.

    Parameters:
        k : wave number (scalar)
        theta : incident angle (in radians)

    Source:
        https://users.math.msu.edu/users/weig/PAPER/p103.pdf
    """

    def __init__(self, option=None):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)
        self.k  = bm.tensor(option.get('k', 1.0))
        self.theta = bm.tensor(option.get('theta', bm.pi/4))
        self.k1 = self.k * bm.cos(self.theta)
        self.k2 = self.k * bm.sin(self.theta)            

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """u(x, y) = exp(i(k1 x + k2 y))"""
        x, y = p[..., 0], p[..., 1]
        phase = self.k1 * x + self.k2 * y
        return bm.exp(1j * phase)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Right-hand side f(x, y) ≡ 0"""
        return bm.zeros_like(p[..., 0], dtype=bm.complex128)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """∇u = i·[k1, k2]·exp(i(k1 x + k2 y))"""
        u_val = self.solution(p)
        grad = 1j * u_val[..., None] * bm.stack([self.k1, self.k2], axis=-1)
        return grad

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Robin boundary data: g = ∂u/∂n + i·k·u
        """
        kappa = 1j * self.k
        grad = self.gradient(p)
        if len(grad.shape) == self.geo_dimension():
            val = bm.sum(grad * n, axis=-1)
        else:
            val = bm.sum(grad * n[:, None, :], axis=-1)
        val += kappa * self.solution(p)
        
        return val

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        """All boundaries use Robin boundary"""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x - 0.0) < atol) |
            (bm.abs(x - 1.0) < atol) |
            (bm.abs(y - 0.0) < atol) |
            (bm.abs(y - 1.0) < atol)
        )
        return on_boundary
