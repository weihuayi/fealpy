from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike

class LshapeData2D:
    """
    2D Stokes problem with stream function formulation on [-1,1]^2 \ [0,1]^2:

        -nu * Δu + ∇p = f,   in Ω
         div u = 0,          in Ω
         u = g,              on ∂Ω
    Exact solution:
        u = (5/2·r^{3/2}·sin(3/2·θ), 5/2·r^{3/2}·cos(3/2·θ))
        p = x^2 + y^2 - 3/2,
        φ = r^{5/2}·sin(5/2·θ)  (stream function)
    Corresponding source f = (2x, 2y), and stream function rhs = 0

    Stream formulation:
        u = curl φ,
        u1 = ∂φ/∂y,
        u2 = -∂φ/∂x
    """
    def viscosity(self) -> float:
        return 0.3
    
    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [-1., 1., -1., 1.]

    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        d = self.domain()
        mesh = TriangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh
    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        # This requires a mesh generator that supports polygonal domains with holes.
        # Define L-shaped polygon as outer square minus inner square.
        return TriangleMesh.from_box(self.domain(), nx=nx, ny=ny, threshold=self.threshold) 

    @cartesian
    def threshold(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (x>=0) & (y>=0)

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r = bm.where(r < 1e-13, 0.0, r)
        theta = bm.atan2(y, x)
        theta = bm.where(theta < 0, theta + 2 * bm.pi, theta)
        u1 = (5/2) * r**(3/2) * bm.sin(3/2 * theta)
        u2 = (5/2) * r**(3/2) * bm.cos(3/2 * theta)
        return bm.stack((u1, u2), axis=-1)   

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return x**2 + y**2 - 3/2

    @cartesian
    def stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r = bm.where(r < 1e-13, 0.0, r)
        theta = bm.atan2(y, x)
        theta = bm.where(theta < 0, theta + 2 * bm.pi, theta)
        return r**(5/2) * bm.sin(5/2 * theta)

    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r = bm.where(r < 1e-13, 0.0, r)
        theta = bm.atan2(y, x)
        theta = bm.where(theta < 0, theta + 2 * bm.pi, theta)
        sin = bm.sin
        cos = bm.cos
        du1_dx = 3.75*r**0.5*sin(0.5*theta)
        du1_dy = 3.75*r**0.5*cos(0.5*theta)
        du2_dx = 3.75*r**0.5*cos(0.5*theta)
        du2_dy = -3.75*r**0.5*sin(0.5*theta)
        return bm.stack([
            bm.stack((du1_dx, du1_dy), axis=-1),
            bm.stack((du2_dx, du2_dy), axis=-1)
        ], axis=-2)  # shape (..., 2, 2)

    @cartesian
    def grad_stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r = bm.where(r < 1e-13, 0.0, r)
        theta = bm.atan2(y, x)
        theta = bm.where(theta < 0, theta + 2 * bm.pi, theta)
        sin = bm.sin
        cos = bm.cos

        dphi_dx = 2.5*r**1.5*sin(1.5*theta) 
        dphi_dy = 2.5*r**1.5*cos(1.5*theta) 
        return bm.stack((dphi_dx, dphi_dy), axis=-1)

    @cartesian
    def hessian_stream_function(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r = bm.where(r < 1e-13, 0.0, r)
        theta = bm.atan2(y, x)
        theta = bm.where(theta < 0, theta + 2 * bm.pi, theta)
        sin = bm.sin
        cos = bm.cos

        d2phi_dx2 = 1.0*(1.25*x**2*sin(2.5*theta) - 7.5*x*y*cos(2.5*theta) - 6.25*y**2*sin(2.5*theta) + 2.5*r**2.0*sin(2.5*theta))/r**(3/2) 
        d2phi_dxdy = (6.25*x**2*cos(2.5*theta) + 7.5*x*y*sin(2.5*theta) - 1.25*y**2*cos(2.5*theta) - 2.5*r**2*cos(2.5*theta))/r**(3/2) 
        d2phi_dy2 =1.0*(-6.25*x**2*sin(2.5*theta) + 7.5*x*y*cos(2.5*theta) + 1.25*y**2*sin(2.5*theta) + 2.5*r**2*sin(2.5*theta))/r**(3/2)
        return bm.stack((d2phi_dx2, d2phi_dxdy, d2phi_dy2), axis=-1)

    @cartesian
    def stream_function_rhs(self, p: TensorLike) -> TensorLike:
        # f = -Δ² φ = 0
        x, y = p[..., 0], p[..., 1]
        return bm.zeros_like(x)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.stack((2 * x, 2 * y), axis=-1)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (bm.abs(x+ 1.0) < atol) | (bm.abs(x - 1.0) < atol) | \
               (bm.abs(y+ 1.0) < atol) | (bm.abs(y - 1.0) < atol) | \
               ((x >= 0.0) & (x < 1.0) & (y > 0.0) & (y <= 1.0))  # cut-out region

 




