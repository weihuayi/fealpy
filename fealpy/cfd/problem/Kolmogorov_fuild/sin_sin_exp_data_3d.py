from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class SinSinExpData3D:
    """
    3D Kolmogorov Flow Example for Incompressible Navier-Stokes:

        ∂u/∂t + u·∇u = -∇p + νΔu
        ∇·u = 0
        (x, y, z) ∈ (0, 1)^3, t > 0

    with exact solution:

        u(x, y, z, t) = (
            sin(2πy) * sin(2πz) * exp(-8π²νt),
            sin(2πz) * sin(2πx) * exp(-8π²νt),
            sin(2πx) * sin(2πy) * exp(-8π²νt)
        )
        p(x, y, z, t) = 0

    The corresponding source term is:

        f(x, y, z, t) = 0

    Periodic boundary conditions are applied in x, y, z directions.
    """

    def __init__(self, nu: float = 0.01, time: float = 0.0):
        self.nu = nu
        self.time = time

    def geo_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # [x0, x1, y0, y1, z0, z1]

    def solution(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        exp_term = bm.exp(-8 * pi**2 * self.nu * self.time)
        u = bm.sin(2 * pi * y) * bm.sin(2 * pi * z) * exp_term
        v = bm.sin(2 * pi * z) * bm.sin(2 * pi * x) * exp_term
        w = bm.sin(2 * pi * x) * bm.sin(2 * pi * y) * exp_term
        return bm.stack([u, v, w], axis=-1)

    def gradient(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        exp_term = bm.exp(-8 * pi**2 * self.nu * self.time)

        # Compute partial derivatives ∂u_i/∂x_j, shape (..., 3, 3)
        # u = [u, v, w]

        # ∂u/∂x = 0
        du_dx = bm.zeros_like(x)
        # ∂u/∂y = 2π cos(2π y) sin(2π z) * exp_term
        du_dy = 2 * pi * bm.cos(2 * pi * y) * bm.sin(2 * pi * z) * exp_term
        # ∂u/∂z = 2π sin(2π y) cos(2π z) * exp_term
        du_dz = 2 * pi * bm.sin(2 * pi * y) * bm.cos(2 * pi * z) * exp_term

        # ∂v/∂x = 2π sin(2π z) cos(2π x) * exp_term
        dv_dx = 2 * pi * bm.sin(2 * pi * z) * bm.cos(2 * pi * x) * exp_term
        # ∂v/∂y = 0
        dv_dy = bm.zeros_like(x)
        # ∂v/∂z = 2π cos(2π z) sin(2π x) * exp_term
        dv_dz = 2 * pi * bm.cos(2 * pi * z) * bm.sin(2 * pi * x) * exp_term

        # ∂w/∂x = 2π cos(2π x) sin(2π y) * exp_term
        dw_dx = 2 * pi * bm.cos(2 * pi * x) * bm.sin(2 * pi * y) * exp_term
        # ∂w/∂y = 2π sin(2π x) cos(2π y) * exp_term
        dw_dy = 2 * pi * bm.sin(2 * pi * x) * bm.cos(2 * pi * y) * exp_term
        # ∂w/∂z = 0
        dw_dz = bm.zeros_like(x)

        grad = bm.stack([
            bm.stack([du_dx, du_dy, du_dz], axis=-1),
            bm.stack([dv_dx, dv_dy, dv_dz], axis=-1),
            bm.stack([dw_dx, dw_dy, dw_dz], axis=-1),
        ], axis=-2)

        return grad

    def source(self, p: TensorLike) -> TensorLike:
        # Zero forcing (homogeneous)
        x = p[..., 0]
        zeros = bm.zeros_like(x)
        return bm.stack([zeros, zeros, zeros], axis=-1)

    def dirichlet(self, p: TensorLike) -> TensorLike:
        # For compatibility; periodic BCs apply so this is not actually used
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        # No Dirichlet boundary for periodic problem
        return bm.zeros_like(p[..., 0], dtype=bm.bool_)