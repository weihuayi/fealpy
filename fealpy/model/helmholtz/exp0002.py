from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..box_domain_mesher import BoxDomainMesher2d
from ...decorator import cartesian

class EXP0002(BoxDomainMesher2d):
    """
    2D Helmholtz problem with complex Robin (impedance-type) boundary condition:

        -Δu(x, y) - k^2·u(x, y) = 0,       (x, y) ∈ (0, 1) x (0, 1)
           ∂u/∂n + i·k·u = g(x, y),        on ∂Ω

    Exact solution:

        u(x, y) = exp(i·β·k·y) * exp(-k·sqrt(β² - 1)·(x + 1))

    This represents an evanescent wave propagating in the y-direction,
    and exponentially decaying in the x-direction.

    Parameters:
        k : wave number (float)
        beta : dimensionless propagation parameter (β > 1)

    Source: 
        https://www.sciencedirect.com/science/article/pii/S0045794917302602#e0010
    """

    def __init__(self, option=None):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)
        self.k  = bm.tensor(option.get('k', 1.0))
        self.beta = bm.tensor(option.get('beta', 1.001))
        self.gamma = bm.sqrt(self.beta**2 - 1.0) # decay rate in x

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        k, beta, gamma = self.k, self.beta, self.gamma
        return bm.exp(1j * beta * k * y) * bm.exp(-k * gamma * (x + 1.0))

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        k, beta, gamma = self.k, self.beta, self.gamma
        u = self.solution(p)
        du_dx = -k * gamma * u
        du_dy = 1j * beta * k * u
        return bm.stack((du_dx, du_dy), axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Right-hand side f(x, y) ≡ 0"""
        return bm.zeros_like(p[..., 0], dtype=bm.complex128)

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
        """Check if point is on boundary ∂Ω to apply Robin condition"""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x - 0.0) < atol) |
            (bm.abs(x - 1.0) < atol) |
            (bm.abs(y - 0.0) < atol) |
            (bm.abs(y - 1.0) < atol)
        )
        return on_boundary



