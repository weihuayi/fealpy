from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..mesher import LshapeMesher


class Exp0004(LshapeMesher):
    """
    L-shaped domain corner singularity problem (2D Poisson):

        -Δu(x, y) = f(x, y),  (x, y) ∈ (-1, 1)^2 \ [0, 1) x (-1, 0]
         u(x, y) = g(x, y),    on ∂Ω

    with the exact solution:

        u(x, y) = r**(2/3) * sin(2/3*theta),
        where r = sqrt(x^2 + y^2), theta = arctan(y, x) % (2 * bm.pi)
    
    The corresponding source term is:

        f(x, y) = 0
    """
    def __init__(self):
        super().__init__()
        
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        r = bm.linalg.norm(p, axis=-1)
        theta = bm.arctan2(y, x) % (2 * bm.pi)
        val = r**(2.0/3) * bm.sin(2.0/3 * theta)
        return val 

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        r = bm.linalg.norm(p, axis=-1)
        theta = bm.arctan2(y, x) % (2 * bm.pi)
        dx = -2/3 * r**(-1.0/3) * bm.sin(1.0/3 * theta)
        dy =  2/3 * r**(-1.0/3) * bm.cos(1.0/3 * theta)
        return bm.stack([dx, dy], axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        val = bm.zeros(len(p), dtype=bm.float64)
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
    
        # 检查是否接近 x=±1 或 y=±1 或 x=0&<0 或 y=0&x>0
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x + 1.) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y + 1.) < atol) |
            ((bm.abs(x - 0.) < atol) & ( y < 0)) |
            ((bm.abs(y - 0.) < atol) & ( x > 0))
        )
        return on_boundary