from typing import Callable, Dict, Literal, Sequence
from ...backend import TensorLike 
from ...backend import backend_manager as bm
from ..boundary_condition import BoundaryCondition, bc_mask, bc_value

class CosCosData2D:
    """
    2D Elliptic equation:

        -∇·(A ∇u(x, y)) + c u(x, y) = f(x, y),  (x, y) ∈ Ω = (0, 1) × (0, 1)
                                  u(x, y) = g(x, y),  on ∂Ω

    with the exact solution:
        u(x, y) = cos(2πx) * cos(2πy)

    where:
        - A(x, y) = [[10, 1], [1, 10]]  (diffusion tensor)
        - c(x, y) = 2                  (reaction coefficient)
        - f(x, y) is computed accordingly
    """

    def __init__(self, bcs: Sequence[BoundaryCondition] = None):
        """
        Initialize the PDE data with boundary conditions.

        Parameters:
            bcs (Sequence[BoundaryCondition]): List of boundary conditions. If None,
                homogeneous Dirichlet condition is imposed on all edges.
        """
        if bcs is None:
            bcs = [
                BoundaryCondition(
                    mask_fn=lambda p: bm.logical_or(
                        (bm.abs(p[:, 0] - 0.0) < 1e-12) |
                        (bm.abs(p[:, 0] - 1.0) < 1e-12) |
                        (bm.abs(p[:, 1] - 0.0) < 1e-12) |
                        (bm.abs(p[:, 1] - 1.0) < 1e-12)
                    ),
                    kind='dirichlet',
                    value_fn=self.solution  # NOTE: must be bound method
                )
            ]
        self.bcs = bcs

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        """
        Return diffusion tensor A(x, y), constant in this example.
        Shape: (..., 2, 2)
        """
        val = bm.array([[10.0, 0.0], [0.0, 10.0]])
        return val 

    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        """
        Return inverse of diffusion tensor A(x, y), constant.
        Shape: (..., 2, 2)
        """
        val = bm.array([[0.1, 0.0], [0.0, 0.1]])  # Approximate inverse
        return val 

    def reaction_coef(self, p: TensorLike) -> TensorLike:
        """
        Return reaction coefficient c(x, y), constant in this case.
        Shape: (...,)
        """
        val = bm.array([2.0])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    def solution(self, p: TensorLike) -> TensorLike:
        """
        Return the exact solution u(x, y) = cos(2πx) * cos(2πy)
        Shape: (...,)
        """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.cos(2*pi*x) * bm.cos(2*pi*y)

    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Return the gradient of the exact solution ∇u(x, y)
        Shape: (..., 2)
        """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.stack((
            -2*pi * bm.sin(2*pi*x) * bm.cos(2*pi*y),
            -2*pi * bm.cos(2*pi*x) * bm.sin(2*pi*y)
        ), axis=-1)

    def flux(self, p: TensorLike) -> TensorLike:
        """
        Return the flux vector -A ∇u
        Shape: (..., 2)
        """
        grad = self.gradient(p)                  # (..., 2)
        A = self.diffusion_coef(p)               # (..., 2, 2)
        return bm.einsum('...ij,...j->...i', A, -grad)

    def source(self, p: TensorLike) -> TensorLike:
        """
        Return the source term f(x, y), computed from -div(A∇u) + c·u
        Shape: (...,)
        """
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        cos = bm.cos
        sin = bm.sin

        # Compute f(x, y) manually
        val = (80*pi**2 + 2)*cos(2*pi*x)*cos(2*pi*y) \
              - 4*pi*sin(2*pi*x)*cos(2*pi*y) \
              - 2*pi*sin(2*pi*y)*cos(2*pi*x)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        """
        Evaluate Dirichlet boundary value at given points.
        """
        return bc_value(p, self.bcs, 'dirichlet')

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """
        Identify whether given points are on the Dirichlet boundary.
        """
        return bc_mask(p, self.bcs, 'dirichlet')

    def neumann(self, p: TensorLike) -> TensorLike:
        """
        Evaluate Neumann boundary value (∇u·n) if defined.
        """
        return bc_value(p, self.bcs, 'neumann')

    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        """
        Identify whether given points are on the Neumann boundary.
        """
        return bc_mask(p, self.bcs, 'neumann')

