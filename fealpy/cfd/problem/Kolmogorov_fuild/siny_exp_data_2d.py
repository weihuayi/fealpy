from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinyExpData2D:
    """
    2D Kolmogorov Flow Example for Incompressible Navier-Stokes:

        ∂u/∂t + u·∇u = -∇p + νΔu
        ∇·u = 0
        (x, y) ∈ (0, 1)², t > 0

    with exact solution:

        u(x, y, t) = ( sin(2πy) * exp(-4π²νt), 0 )
        p(x, y, t) = 0

    The corresponding source term is:

        f(x, y, t) = 0

    Periodic boundary conditions are applied in both x and y directions.
    """

    def __init__(self, nu: float = 0.01, time: float = 0.0):
        self.nu = nu
        self.time = time

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]  # [x0, x1, y0, y1]

    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        u = bm.sin(2 * pi * y) * bm.exp(-4 * pi**2 * self.nu * self.time)
        v = bm.zeros_like(x)
        return bm.stack([u, v], axis=-1)

    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        u_y = 2 * pi * bm.cos(2 * pi * y) * bm.exp(-4 * pi**2 * self.nu * self.time)
        # Gradients: du/dx = 0, du/dy = u_y; dv/dx = 0, dv/dy = 0
        # Output shape: (..., 2, 2), where gradient[..., i, j] = ∂u_i/∂x_j
        # u = [u, v], ∂u_i/∂x_j
        zeros = bm.zeros_like(x)
        grad = bm.stack([
            bm.stack([zeros, u_y], axis=-1),   # ∇u
            bm.stack([zeros, zeros], axis=-1)  # ∇v
        ], axis=-2)
        return grad

    def source(self, p: TensorLike) -> TensorLike:
        # Zero forcing (homogeneous)
        x = p[..., 0]
        return bm.stack([bm.zeros_like(x), bm.zeros_like(x)], axis=-1)

    def dirichlet(self, p: TensorLike) -> TensorLike:
        # For compatibility; periodic BCs apply so this is not actually used
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        # No Dirichlet boundary for periodic problem
        return bm.zeros_like(p[..., 0], dtype=bm.bool_)