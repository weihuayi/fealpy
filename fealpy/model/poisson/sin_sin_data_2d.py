
from typing import Sequence
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
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return bm.sin(pi * x) * bm.sin(pi * y)

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        du_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        return bm.stack([du_dx, du_dy], axis=-1)

    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return 2 * pi**2 * bm.sin(pi * x) * bm.sin(pi * y)

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """
        Check if a point is on the Dirichlet boundary (x = 0 or 1) or (y = 0 or 1).
        
        Args:
            p (TensorLike): Input points, shape (n_points, 2).
        
        Returns:
            TensorLike: Boolean tensor indicating whether each point is on the boundary.
        """
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # 绝对误差容限
    
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x + 1.) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y + 1.) < atol)
        )
        return on_boundary 

