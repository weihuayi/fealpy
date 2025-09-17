from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class ExpData2D:
    """
    2D Darcy-Forchheimer problem on Ω = [0,1]×[0,1]:

        PDE: μ u + β |u| u + ∇p = f    in Ω
             ∇·u = g                in Ω
             u · n = 0              on ∂Ω

    Exact fields:
        u(x,y) = (-y x (1 - x), x y (1 - y))^T
        p(x,y) = exp(x(1-x) y(1-y)) - 1
        g(x,y) = x - y

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
        return [0.0, 1.0, 0.0, 1.0]

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Exact velocity u = (-y x (1-x), x y (1-y))."""
        x, y = p[..., 0], p[..., 1]
        u_x = -y * x * (1 - x)
        u_y =  x * y * (1 - y)
        return bm.stack((u_x, u_y), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Pressure p = exp(x(1-x)y(1-y)) - 1."""
        x, y = p[..., 0], p[..., 1]
        return bm.exp(x * (1 - x) * y * (1 - y)) - 1

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure ∇p."""
        x, y = p[..., 0], p[..., 1]
        psi = x * (1 - x) * y * (1 - y)
        exp_psi = bm.exp(psi)
        dpdx = exp_psi * (1 - 2*x) * y * (1 - y)
        dpdy = exp_psi * x * (1 - x) * (1 - 2*y)
        return bm.stack((dpdx, dpdy), axis=-1)

    @cartesian
    def g(self, p: TensorLike) -> TensorLike:
        """Divergence ∇·u = g(x,y)."""
        x, y = p[..., 0], p[..., 1]
        return x - y

    @cartesian
    def norm_u(self, p: TensorLike) -> TensorLike:
        """Compute |u| for nonlinear term."""
        u = self.velocity(p)
        return bm.sqrt(bm.sum(u * u, axis=-1))

    @cartesian
    def f(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = μ u + β |u| u + ∇p."""
        u = self.velocity(p)
        m = self.mu / (self.rho * self.k) + self.beta * self.norm_u(p)
        return m[..., None] * u + self.grad_pressure(p)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Neumann boundary on all edges (u·n=0)."""
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
        return bm.sum(u * n, axis=-1)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """No Dirichlet boundary; returns False."""
        return bm.zeros_like(p[..., 0], dtype=bm.bool)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet velocity."""
        return self.velocity(p)

    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet pressure."""
        return self.pressure(p)