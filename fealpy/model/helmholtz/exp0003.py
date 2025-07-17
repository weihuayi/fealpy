from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..mesher import BoxMesher2d
from ...decorator import cartesian

class Exp0003(BoxMesher2d):
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

    def set(self, k: float=1.0, theta: float= bm.pi/4):
        self.k = k
        self.theta = theta
        self.k1 = k * bm.cos(theta)
        self.k2 = k * bm.sin(theta)       

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]
    
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
        """Robin boundary value: ∂u/∂n + iku = g(x)"""
        u_val = self.solution(p)
        grad_u = self.gradient(p)
        dndu = bm.sum(grad_u * n[:, None, :], axis=-1)  # ∇u · n
        return dndu + 1j * self.k * u_val

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
