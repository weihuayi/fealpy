from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class PostData5:
    """
    2D exponential-vortex Darcy-Forchheimer problem on Ω = [-1,1]×[-1,1]:

        PDE: μ u + β |u| u + ∇p = f    in Ω
             ∇·u = 0                in Ω
             u · n = 0              on ∂Ω

    Exact solution:
        u(x,y) = (e^x sin(y), e^x cos(y))^T
        p(x,y) = (x + y)/(x^2 + y^2 + 0.02)
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
        """Exact velocity u = (e^x sin(y), e^x cos(y))."""
        x, y = p[...,0], p[...,1]
        ex = bm.exp(x)
        return bm.stack((ex * bm.sin(y), ex * bm.cos(y)), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Pressure p = (x+y)/(x^2+y^2+0.02)."""
        x, y = p[...,0], p[...,1]
        denom = x**2 + y**2 + 0.02
        return (x + y) / denom

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure ∇p."""
        x, y = p[...,0], p[...,1]
        denom = x**2 + y**2 + 0.02
        factor = 1/(denom**2)
        dpdx = (denom - 2*x*(x+y)) * factor
        dpdy = (denom - 2*y*(x+y)) * factor
        return bm.stack((dpdx, dpdy), axis=-1)

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
