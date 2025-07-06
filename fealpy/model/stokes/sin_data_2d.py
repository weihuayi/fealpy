from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinData2D:
    """
    2D Stokes problem with stream function formulation on [0,1]^2:

        -nu * Δu + ∇p = f,   in Ω
         div u = 0,          in Ω
         u = g,              on ∂Ω

    Exact solution:
        u = (sin(x+y), -sin(x+y)),
        p = x^2 + y^2 - 8/3,
        phi = -cos(x+y)  (stream function)
    Corresponding source f computed from f = -nu Δu + ∇p
    Homogeneous Dirichlet b.c. for velocity.

    Stream formulation:
        u = curl φ,
        u1 = ∂φ/∂y,
        u2 = -∂φ/∂x
    """
    def viscosity(self) -> float:
        return 0.3
    
    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0., 1., 0., 1.]

    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        d = self.domain()
        mesh = TriangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u1 = bm.sin(x + y)
        u2 = -bm.sin(x + y)
        return bm.stack((u1, u2), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return x**2 + y**2 - 8/3

    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        du1_dx = bm.cos(x + y)
        du1_dy = bm.cos(x + y)
        du2_dx = -bm.cos(x + y)
        du2_dy = -bm.cos(x + y)
        return bm.stack([
            bm.stack((du1_dx, du1_dy), axis=-1),
            bm.stack((du2_dx, du2_dy), axis=-1)
        ], axis=-2)  # shape (..., 2, 2)

    @cartesian
    def stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return -bm.cos(x + y)

    @cartesian
    def grad_stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        dphi_dx = bm.sin(x + y)
        dphi_dy = bm.sin(x + y)
        return bm.stack((dphi_dx, dphi_dy), axis=-1)

    @cartesian
    def hessian_stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        d2phi_dx2 = bm.cos(x + y)
        d2phi_dxdy = bm.cos(x + y)
        d2phi_dydx = bm.cos(x + y)
        d2phi_dy2 = bm.cos(x + y)
        return bm.stack((d2phi_dx2, d2phi_dxdy, d2phi_dy2), axis=-1)

    @cartesian
    def stream_function_source(self, p: TensorLike) -> TensorLike:
        # f = -Δ² φ = -Δ(Δφ) = -Δ(2cos(x+y)) = -4cos(x+y)
        x, y = p[..., 0], p[..., 1]
        return -4 * bm.cos(x + y)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        # f = -nu Δu + ∇p; here Δu = 0 since u is harmonic sin(x+y)
        x, y = p[..., 0], p[..., 1]
        fx = 2*x - 2*bm.sin(x + y)
        fy = 2*y + 2*bm.sin(x + y)
        return bm.stack((fx, fy), axis=-1)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        # velocity Dirichlet boundary condition equals exact solution
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x - 0.0) < atol) | (bm.abs(x - 1.0) < atol) |
            (bm.abs(y - 0.0) < atol) | (bm.abs(y - 1.0) < atol)
        )
        return on_boundary


