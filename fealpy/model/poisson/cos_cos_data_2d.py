from typing import Sequence
from ...decorator import cartesian
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
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [-1., 1., -1., 1.]
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = bm.cos(pi*x)*bm.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -pi*bm.sin(pi*x)*bm.cos(pi*y),
            -pi*bm.cos(pi*x)*bm.sin(pi*y)), axis=-1)
        return val # val.shape == p.shape

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        val = 2*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)
        return val

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

