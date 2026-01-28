from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d


class Exp0003(BoxMesher2d):
    """
    2D Darcy–Forchheimer test problem on Ω = [0,1] × [0,1].

    Strong form:
        μ u + β |u| u + ∇p = f    in Ω
        ∇·u = 0                   in Ω
        u · n = g_N                 on ∂Ω

    Manufactured (exact) solution used to build RHS and boundary data:
        velocity: u(x,y) = (0.25*(x+1)**2, -0.5*(x+1)*(y+1))^T
        pressure: p(x,y)  = x**3 + y**3
        g(x,y) = 0
        g_N = 1+y, x= 1;
        1-y, x=-1;
         x-1, y=1;
          -x-1, y=-1;

    Forcing term:
        f(x,y) = μ u + β |u| u + ∇p

    Model parameters (set in __init__):
        mu   = 1
        beta = 30
        rho  = 1.0
        tol  = 1e-12
    """

    def __init__(self):
        # physical / numerical parameters
        self.box = [-1, 1, -1, 1.0]
        super().__init__(self.box)
        self.mu = 1
        self.beta = 0
        self.rho = 1.0

    def geo_dimension(self) -> int:
        """Return geometric dimension (2 means 2D)."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return computational domain [xmin, xmax, ymin, ymax]."""
        return [-1, 1.0, -1, 1.0]

    @cartesian
    def g(self, p: TensorLike) -> TensorLike:
        """Source term g(x,y). Here it is zero (no volumetric source)."""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """
        Exact velocity field u(x,y).

        Note: this example uses a polynomial velocity (manufactured solution).
        """
        x, y = p[..., 0], p[..., 1]
        u1 = 0.25 * (x + 1) ** 2
        u2 = -0.5 * (x + 1) * (y + 1)
        return bm.stack((u1, u2), axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Exact pressure p(x,y) = x^3 + y^3."""
        x, y = p[..., 0], p[..., 1]
        return x ** 3 + y ** 3

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of the pressure ∇p = (∂p/∂x, ∂p/∂y)."""
        x, y = p[..., 0], p[..., 1]
        dpdx = 3 * x ** 2
        dpdy = 3 * y ** 2
        return bm.stack((dpdx, dpdy), axis=-1)

    @cartesian
    def norm_u(self, p: TensorLike) -> TensorLike:
        """Compute |u| = sqrt(u_x^2 + u_y^2)."""
        u = self.velocity(p)
        return bm.sqrt(bm.sum(u * u, axis=-1))

    @cartesian
    def f(self, p: TensorLike) -> TensorLike:
        """Right-hand side f = μ u + β |u| u + ∇p (constructed from the exact solution)."""
        u = self.velocity(p)
        m = self.mu + self.beta * self.norm_u(p)
        return m[..., None] * u + self.grad_pressure(p)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:

        x, y = p[..., 0], p[..., 1]
        xmin, xmax, ymin, ymax = self.box
        left   = bm.abs(x - xmin) < self.tol
        right  = bm.abs(x - xmax) < self.tol
        bottom = bm.abs(y - ymin) < self.tol
        top    = bm.abs(y - ymax) < self.tol
        return left | right | bottom | top


    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """Neumann boundary value: the normal flux u·n (using the exact velocity)."""
        u = self.velocity(p)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        return bm.einsum("fqd,fqd->fq", u, n)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator for Dirichlet boundary. None here (no Dirichlet conditions used)."""
        return None

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet velocity (provided for completeness): returns the exact velocity."""
        return self.velocity(p)

    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Unused Dirichlet pressure (provided for completeness): returns the exact pressure."""
        return self.pressure(p)
