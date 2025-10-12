from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    2D rotational-arctan Darcy-Forchheimer problem on Ω = [0,1]×[0,1]:

        PDE: μ u + β |u| u + ∇p = f    in Ω
             ∇·u = 0                in Ω
             u · n = 0              on ∂Ω

    Exact solution:
        u(x,y) = (-y, x)^T
        p(x,y) = 2/π arctan(10(x+y-1))
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
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
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
    def g(self, p: TensorLike) -> TensorLike:
        """Source term g(x,y) = 0."""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Exact rotation velocity u = (-y, x)."""
        x, y = p[..., 0], p[..., 1]
        return bm.stack((-y, x), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Pressure p = 2/π arctan(10(x+y-1))."""
        arg = 10 * (p[...,0] + p[...,1] - 1)
        return 2/bm.pi * bm.atan(arg)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure ∇p."""
        x, y = p[..., 0], p[..., 1]
        denom = 1 + (10*(x+y-1))**2
        factor = 20/(bm.pi * denom)
        return bm.stack((factor, factor), axis=-1)

    @cartesian
    def norm_u(self, p: TensorLike) -> TensorLike:
        """Compute |u| = sqrt(x^2+y^2)."""
        u = self.velocity(p)
        return bm.sqrt(bm.sum(u*u, axis=-1))

    @cartesian
    def f(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = μ u + β |u| u + ∇p."""
        u = self.velocity(p)
        m = self.mu / (self.rho * self.k) + self.beta * self.norm_u(p)
        return m[..., None] * u + self.grad_pressure(p)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Neumann boundary (u·n=0) on ∂Ω."""
        x, y = p[...,0], p[...,1]
        left  = bm.abs(x - 0.0) < self.tol
        right = bm.abs(x - 1.0) < self.tol
        bottom= bm.abs(y - 0.0) < self.tol
        top   = bm.abs(y - 1.0) < self.tol
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
