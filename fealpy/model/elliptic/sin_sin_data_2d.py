from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinSinData2D:
    """
    2D Elliptic equation with constant diagonal diffusion, advection, and reaction:

        -∇·(A ∇u(x, y)) + b·∇u(x, y) + c u(x, y) = f(x, y),  (x, y) ∈ (0, 1) x (0, 1)
                                  u(x, y) = g(x, y),          on ∂Ω

    Exact solution:
        u(x, y) = sin(πx) * sin(πy)

    Coefficients:
        A = [[2, 0], [0, 3]]  (diagonal diffusion matrix)
        b = [1, -1]           (constant advection vector)
        c = 4                 (constant reaction coefficient)
        f(x, y) = (5π² + 4)sin(πx)sin(πy) + πcos(πx)sin(πy) - πsin(πx)cos(πy)
        g(x, y) = 0           (Dirichlet boundary condition)
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    def diffusion_coef(self) -> TensorLike:
        """Constant diagonal diffusion tensor (shape: (2, 2)."""
        A = bm.array([[2.0, 0.0], [0.0, 3.0]])
        return A 

    def diffusion_coef_inv(self) -> TensorLike:
        """Inverse of diffusion tensor (shape: (2, 2) ."""
        A_inv = bm.array([[0.5, 0.0], [0.0, 1.0/3.0]])
        return A_inv 

    def convection_coef(self) -> TensorLike:
        """Constant advection vector (shape: (2,))."""
        b = bm.array([1.0, -1.0])
        return b

    def reaction_coef(self) -> TensorLike:
        """Constant reaction coefficient."""
        return bm.tensor([4.0])

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Exact solution: u(x, y) = sin(πx) sin(πy)."""
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Gradient of exact solution (shape: (..., 2))."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.stack([
            pi * bm.cos(pi * x) * bm.sin(pi * y),  # ∂u/∂x
            pi * bm.sin(pi * x) * bm.cos(pi * y)   # ∂u/∂y
        ], axis=-1)

    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        """Flux vector: -A ∇u (shape: (..., 2))."""
        grad = self.gradient(p)
        A = self.diffusion_coef()
        return -bm.einsum('...ij,...j->...i', A, grad)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Source term f(x, y) derived from PDE (shape: (...,))."""
        
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        term1 = (5*pi**2 + 4) * sin(pi*x) * sin(pi*y)
        term2 = pi * cos(pi*x) * sin(pi*y)
        term3 = -pi * sin(pi*x) * cos(pi*y)
        return term1 + term2 + term3

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition."""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (
            (bm.abs(x) < atol) | (bm.abs(x - 1) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y - 1) < atol)
        )

