from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinData1D:
    """
    Darcy-Forchheimer 1D test PDE:

        (μ/k + β·ρ·|u|)·u + ∂p/∂x = g(x)
        ∂u/∂x = f(x)
        μ = 2.0, k = 4.0, β = 5.0, ρ = 1.0

    With exact solutions:

        p(x) = x - x²
        u(x) = sin(π x)

    Source and forcing terms:

        g(x) = (μ/k + β·ρ·|sin(π x)|)·sin(π x) - 2 x + 1
        f(x) = cos(π x)
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension (1D)."""
        return 1

    def domain(self) -> Sequence[float]:
        """Computational domain [xmin, xmax]."""
        return [0.0, 1.0]

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Exact pressure: p(x) = x - x^2"""
        return p - p**2

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Exact velocity: u(x) = sin(pi * x)"""
        pi = bm.pi
        return bm.sin(pi * p)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        """Gradient of pressure: dp/dx = 1 - 2*x"""
        return 1 - 2 * p

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Right-hand side source f(x) = pi * cos(pi * x)"""
        pi = bm.pi
        return pi * bm.cos(pi * p)

    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        """Nonlinear flux term g(u,p) = (mu/k + beta*rho*|u|)*u + dp/dx"""
        u = self.velocity(p)
        dpdx = self.grad_pressure(p)
        norm_u = bm.abs(u)
        return (1 / 2 + 5 * norm_u) * u + dpdx

    @cartesian
    def neumann(self, p: TensorLike) -> TensorLike:
        """Neumann boundary: use velocity as flux at boundary."""
        return self.velocity(p)
