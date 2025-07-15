from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class SinSinData2D:
    """
    2D Quasilinear Poisson-type problem:

        -div(mu(|∇u|) ∇u) = f(x, y),  in Ω = (-1,1)²
                          u = 0       on ∂Ω

    with exact solution:
        u(x, y) = sin(πx) * sin(πy)

    and nonlinear diffusion coefficient:
        mu(|∇u|) = 2 + 1 / (1 + |∇u|²)
    """
    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.sin(pi * x) * bm.sin(pi * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.stack((
            pi * bm.cos(pi * x) * bm.sin(pi * y),
            pi * bm.sin(pi * x) * bm.cos(pi * y)
        ), axis=-1)

    @cartesian
    def nonlinear_coeff(self, p: TensorLike) -> TensorLike:
        """Nonlinear coefficient function μ(|∇u|)"""
        grad = self.gradient(p)
        grad_norm_sq = bm.sum(grad**2, axis=-1)
        return 2.0 + 1.0 / (1.0 + grad_norm_sq)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = -div(μ(|∇u|) ∇u)"""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        cos = bm.cos
        sin = bm.sin
        return 2*pi**2*(pi**2*(cos(pi*x)**2*cos(2*pi*y) + cos(2*pi*x)*cos(pi*y)**2) + (pi**2*sin(pi*x)**2*cos(pi*y)**2 + pi**2*sin(pi*y)**2*cos(pi*x)**2 + 1)*(2*pi**2*sin(pi*x)**2*cos(pi*y)**2 + 2*pi**2*sin(pi*y)**2*cos(pi*x)**2 + 3))*sin(pi*x)*sin(pi*y)/(pi**2*sin(pi*x)**2*cos(pi*y)**2 + pi**2*sin(pi*y)**2*cos(pi*x)**2 + 1)**2 

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (bm.abs(x) < atol) | (bm.abs(x - 1.0) < atol) | \
               (bm.abs(y) < atol) | (bm.abs(y - 1.0) < atol)
