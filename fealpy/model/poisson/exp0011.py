from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0011(BoxMesher2d):
    """
    2D Poisson problem:
    
        -Δu(x, y) = f(x, y),  (x, y) ∈ (0, 1)^2
         u(x, y) = g(x, y),   on ∂Ω

    with the exact solution:

        u(x, y) = exp(-(x² + y²)/2)

    The corresponding source term is:

        f(x, y) = -(x² + y² - 2) * exp(-(x² + y²)/2)

    Dirichlet boundary conditions are applied on all edges using the exact solution.
    """
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0] 
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution: u(x, y) = exp(-(x² + y²)/2)"""
        x, y = p[..., 0], p[..., 1]
        return bm.exp(-0.5 * (x**2 + y**2))

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution: ∇u = (-x u, -y u)"""
        x, y = p[..., 0], p[..., 1]
        u = self.solution(p)
        du_dx = -x * u
        du_dy = -y * u
        return bm.stack([du_dx, du_dy], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term: f = -(x² + y² - 2) * exp(-(x² + y²)/2)"""
        x, y = p[..., 0], p[..., 1]
        u = self.solution(p)
        return -(x**2 + y**2 - 2) * u

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition: u = g (exact solution)"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # Absolute tolerance
    
        # Check if close to x=0, x=1, y=0, or y=1
        on_boundary = (
            (bm.abs(x) < atol) | (bm.abs(x - 1.) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y - 1.) < atol)
        )
        return on_boundary 