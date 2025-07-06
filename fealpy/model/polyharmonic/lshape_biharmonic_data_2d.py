from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike


class LShapeBiharmonicData2D:
    """
    2D biharmonic problem on L-shaped domain:
    
    Δ²φ = h in Ω = [0,2]^2 \ [0,1]^2,
    with boundary conditions :
      ∂φ/∂n = h_1,  φ = h_2.
    
    Exact solution in polar coordinates (r, θ):
      φ(r, θ) = r^(3/2) * sin(3θ/2)
    
    Viscosity parameter nu = 0.01 (if needed for further modeling).
    """
    
    def geo_dimension(self) -> int:
        """Return geometric dimension of the domain."""
        return 2
    
    def domain(self):
        """Return bounding box of the computational domain."""
        # Box large enough to contain L shape; actual domain excludes [-1,1]^2
        return [-1.0, 1.0, -1.0, 1.0]
    
    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        # This requires a mesh generator that supports polygonal domains with holes.
        # Define L-shaped polygon as outer square minus inner square.
        return TriangleMesh.from_box(self.domain(), nx=nx, ny=ny, threshold=self.threshold) 

    @cartesian
    def threshold(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (x>=1) & (y>=1)
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Exact solution φ(r, θ) = r^{3/2} sin(3θ/2)."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] +=2*bm.pi
        return r**(3/2) * bm.sin(3/2 * theta)
    
    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Gradient ∇φ in Cartesian coordinates."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2 * bm.pi
        sin = bm.sin
        cos = bm.cos
        u_x = 3*x*sin(3*theta/2)/(2*(x**2 + y**2)**(1/4)) - 3*y*cos(3*theta/2)/(2*(x**2 + y**2)**(1/4))
        u_y = 3*x*cos(3*theta/2)/(2*(x**2 + y**2)**(1/4)) + 3*y*sin(3*theta/2)/(2*(x**2 + y**2)**(1/4))
        u_x = bm.where(r>1e-14, u_x, 0)
        u_y = bm.where(r>1e-14, u_y, 0)
        return bm.stack((u_x, u_y), axis=-1)
    
    @cartesian
    def hessian(self, p: TensorLike) -> TensorLike:
        """Hessian matrix components of φ."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2 * bm.pi
        sin = bm.sin
        cos = bm.cos
        u_xx = 3*(x**2*sin(3*theta/2) - 2*x*y*cos(3*theta/2) - y**2*sin(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_xy = 3*(x**2*cos(3*theta/2) + 2*x*y*sin(3*theta/2) - y**2*cos(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_yy = 3*(-x**2*sin(3*theta/2) + 2*x*y*cos(3*theta/2) + y**2*sin(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_xx = bm.where(r>1e-14, u_xx, 0)
        u_xy = bm.where(r>1e-14, u_xy, 0)
        u_yy = bm.where(r>1e-14, u_yy, 0)
        return bm.stack((u_xx, u_xy, u_yy), axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        return bm.zeros(p.shape[0])
    
    def get_flist(self) -> Sequence[TensorLike]:
        """Return list of functions for solution, gradient, hessian."""
        return [self.solution, self.gradient, self.hessian]
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition on ∂Ω: φ."""
        return self.solution(p)
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (bm.abs(x ) < atol) | (bm.abs(x - 2.0) < atol) | \
               (bm.abs(y ) < atol) | (bm.abs(y - 2.0) < atol) | \
               ((x >= 1.0) & (x < 2.0) & (y > 1.0) & (y <= 2.0))  # cut-out region

