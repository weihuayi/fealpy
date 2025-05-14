
from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike


class CosCosData2D():
    """
    2D Poisson problem:
    
        -Δu(x, y) = f(x, y),  (x, y) ∈ (-1, 1) x (-1, 1)
         u(x, y) = g(x, y),    on ∂Ω

    with the exact solution:

        u(x, y) = cos(πx)·cos(πy)

    The corresponding source term is:

        f(x, y) = 2π²·cos(πx)·cos(πy)
    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [-1., 1., -1., 1.]

    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = bm.cos(pi*x)*bm.cos(pi*y)
        return val # val.shape == x.shape

    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -pi*bm.sin(pi*x)*bm.cos(pi*y),
            -pi*bm.cos(pi*x)*bm.sin(pi*y)), axis=-1)
        return val # val.shape == p.shape

    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = 2*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)
    
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """
        Check if a point is on the Dirichlet boundary (x = ±1 or y = ±1).
        
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

