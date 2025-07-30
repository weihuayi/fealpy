from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d


class Exp0003(BoxMesher2d):
    """
    2D Darcy-Forchheimer problem on Ω = [0,1]×[0,1]:

        PDE: μ u + β |u| u + ∇p = f    in Ω
             ∇·u = 0                in Ω
             u · n = 0              on ∂Ω

    Exact solution:
        u(x,y) = (sin(π x) cos(π y), -cos(π x) sin(π y))^T
        p(x,y) = cos(π x) cos(π y)
        g(x,y) = 0

    Forcing term:
        f(x,y) = μ u + β |u| u + ∇p

    Parameters:
        μ = 2.0
        β = 5.0
        tol = 1e-12
    """
    def __init__(self):
        # physical parameters
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
        self.mu = 1
        self.beta = 30
        self.rho = 1.0
        self.tol = 1e-12

    def geo_dimension(self) -> int:
        """Return geometric dimension (2D)."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    @cartesian
    def g(self, p: TensorLike) -> TensorLike:
        """Source term g(x,y) = 0."""
        return -bm.zeros_like(p[..., 0])

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Exact velocity u = (sin(πx)cos(πy), -cos(πx)sin(πy))."""
        x, y = p[..., 0], p[..., 1]
        x = p[..., 0]
        y = p[..., 1]
        u1 = 0.25*(x+1)**2 
        u2 = -0.5*(x+1)*(y+1)
        return bm.stack((u1, u2), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Exact pressure p = cos(πx)cos(πy)."""
        x, y = p[..., 0], p[..., 1]
        return x**3 + y**3

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure ∇p."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        dpdx = 3*x**2
        dpdy = 3*y**2
        return bm.stack((dpdx, dpdy), axis=-1)

    @cartesian
    def norm_u(self, p: TensorLike) -> TensorLike:
        """Compute |u| = sqrt(u_x^2 + u_y^2)."""
        u = self.velocity(p)
        return bm.sqrt(bm.sum(u * u, axis=-1))

    @cartesian
    def f(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = μ u + β |u| u + ∇p."""
        u = self.velocity(p)
        m = self.mu + self.beta * self.norm_u(p)
        return m[..., None] * u + self.grad_pressure(p)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Neumann boundary (u·n=0) on ∂Ω."""
        x, y = p[..., 0], p[..., 1]
        left   = bm.abs(x - 0.0) < self.tol
        right  = bm.abs(x - 1.0) < self.tol
        bottom = bm.abs(y - 0.0) < self.tol
        top    = bm.abs(y - 1.0) < self.tol
        return left | right | bottom | top

    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """Neumann boundary condition: u·n."""
        u = self.velocity(p)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        return bm.einsum("fqd,fqd->fq", u, n)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Dirichlet boundary: none specified."""
        #return bm.zeros_like(p[..., 0], dtype=bm.bool)
        return None

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet velocity."""
        return self.velocity(p)

    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet pressure."""
        return self.pressure(p)
