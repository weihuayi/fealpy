from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class PostData3:
    """
    2D rotational-radial Darcy-Forchheimer problem on Ω = [-1,1]×[-1,1]:

        PDE: μ u + β |u| u + ∇p = f    in Ω
             ∇·u = 0                in Ω
             u · n = 0              on ∂Ω

    Exact solution:
        u(x,y) = (y/(x^2+y^2+0.02), -x/(x^2+y^2+0.02))^T
        p(x,y) = sqrt(x^2 + y^2 + 0.02)
        g(x,y) = 0

    Forcing term:
        f(x,y) = μ u + β |u| u + ∇p

    Parameters:
        μ = 2.0
        k = 4.0
        ρ = 1.0
        β = 5.0
        tol = 1e-12
    """
    def __init__(self):
        # physical parameters
        self.mu = 2.0
        self.k = 4.0
        self.rho = 1.0
        self.beta = 5.0
        self.tol = 1e-12

    def geo_dimension(self) -> int:
        """Return geometric dimension (2D)."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return computational domain [xmin, xmax, ymin, ymax]."""
        return [-1.0, 1.0, -1.0, 1.0]

    @cartesian
    def g(self, p: TensorLike) -> TensorLike:
        """Source term g(x,y) = 0."""
        return bm.zeros_like(p[...,0])

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Exact velocity u = (y/(r2), -x/(r2))."""
        x, y = p[...,0], p[...,1]
        denom = x**2 + y**2 + 0.02
        return bm.stack((y/denom, -x/denom), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Exact pressure p = sqrt(x^2 + y^2 + 0.02)."""
        x, y = p[...,0], p[...,1]
        return bm.sqrt(x**2 + y**2 + 0.02)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure ∇p."""
        x, y = p[...,0], p[...,1]
        denom = bm.sqrt(x**2 + y**2 + 0.02)
        return bm.stack((x/denom, y/denom), axis=-1)

    @cartesian
    def f(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = μ u + β |u| u + ∇p."""
        u = self.velocity(p)
        norm_u = bm.sqrt(bm.sum(u*u, axis=-1))
        m = self.mu / (self.rho * self.k) + self.beta * norm_u
        return m[...,None] * u + self.grad_pressure(p)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Neumann boundary (u·n=0) on ∂Ω."""
        x, y = p[...,0], p[...,1]
        left   = bm.abs(x + 1.0) < self.tol
        right  = bm.abs(x - 1.0) < self.tol
        bottom = bm.abs(y + 1.0) < self.tol
        top    = bm.abs(y - 1.0) < self.tol
        return left | right | bottom | top

    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """Neumann boundary condition: u·n."""
        u = self.velocity(p)
        return bm.sum(u * n, axis=-1)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """No Dirichlet boundary; returns False."""
        return bm.zeros_like(p[...,0], dtype=bm.bool)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet velocity."""
        return self.velocity(p)

    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet pressure."""
        return self.pressure(p)
