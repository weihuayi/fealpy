from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike 
from ...backend import backend_manager as bm


class CosCosData2D:
    """
    2D Elliptic equation:

        -∇·(A ∇u(x, y)) = f(x, y),  (x, y) ∈ Ω = (0, 1) x (0, 1)
                                  u(x, y) = g(x, y),  on ∂Ω

    with the exact solution:
        u(x, y) = cos(2πx) * cos(2πy)

    where:
        A(x, y) = [[10, 0], [0, 10]]  (diffusion tensor)
        f(x, y) = 80π²cos(2πx)cos(2πy)-8π²sin(2πx)sin(2πy)
    """


    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    def diffusion_coef(self) -> TensorLike:
        """
        Return diffusion tensor A(x, y), constant in this example, Shape: (2, 2).
        """
        val = bm.array([[10.0, 0], [0, 10.0]])
        return val 

    def diffusion_coef_inv(self) -> TensorLike:
        """
        Return inverse of diffusion tensor A(x, y), constant, Shape: (2, 2).
        """
        val = bm.array([[10, 0.0], [0, 10]]) / 100  # Approximate inverse
        return val 

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        Return the exact solution u(x, y) = cos(2πx) * cos(2πy), Shape: (..., ).
        """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.cos(2*pi*x) * bm.cos(2*pi*y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Return the gradient of the exact solution ∇u(x, y), Shape: (..., 2).
        """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.stack((
            -2*pi * bm.sin(2*pi*x) * bm.cos(2*pi*y),
            -2*pi * bm.cos(2*pi*x) * bm.sin(2*pi*y)
        ), axis=-1)

    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        """
        Return the flux vector -A ∇u,  Shape: (..., 2).
        """
        grad = self.gradient(p)                  # (..., 2)
        A = self.diffusion_coef()               # (..., 2, 2)
        return -bm.einsum('...ij,...j->...i', A, -grad)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Return the source term f(x, y)"""
        x, y = p[..., 0], p[..., 1]
        term1 = 80 * (bm.pi**2) * bm.cos(2 * bm.pi * x) * bm.cos(2 * bm.pi * y)
    
        return term1

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

