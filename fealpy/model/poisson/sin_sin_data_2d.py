from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class SinSinData2D:
    """
    2D Poisson problem:
    
        -Δu(x, y) = f(x, y),  (x, y) ∈ (0, 1) x (0, 1)
         u(x, y) = 0,         on ∂Ω

    with the exact solution:

        u(x, y) = sin(πx)·sin(πy)

    The corresponding source term is:

        f(x, y) = 2·π²·sin(πx)·sin(πy)

    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return bm.sin(pi * x) * bm.sin(pi * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        du_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        return bm.stack([du_dx, du_dy], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return 2 * pi**2 * bm.sin(pi * x) * bm.sin(pi * y)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # 绝对误差容限
    
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x + 1.) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y + 1.) < atol)
        )
        return on_boundary 

