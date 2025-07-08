from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..box_domain_mesher import BoxDomainMesher2d
from ...decorator import cartesian

class SinSinData2D(BoxDomainMesher2d):
    """
    2D Helmholtz problem with homogeneous Dirichlet boundary condition:

        -Δu(x, y) - k^2·u(x, y) = f(x, y),  (x, y) ∈ (0, 1) x (0, 1)
                           u(x, y) = 0,    on ∂Ω

    with the exact solution:

        u(x, y) = sin(kπx)·sin(kπy)

    The corresponding source term is:

        f(x, y) = k^2·(2π^2 - 1)·sin(kπx)·sin(kπy)

    Parameter:
        k : wave number (scalar)
    
    Source:
        https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html
    """

    def set(self, k: float = 1.0):
        self.k = k

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u(x, y) = sin(kπx)·sin(kπy)"""
        x, y = p[..., 0], p[..., 1]
        kπ = bm.pi * self.k
        return bm.sin(kπ * x) * bm.sin(kπ * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        kπ = bm.pi * self.k
        val = bm.stack((
            kπ * bm.cos(kπ * x) * bm.sin(kπ * y),
            kπ * bm.sin(kπ * x) * bm.cos(kπ * y)
        ), axis=-1)
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term f(x, y) = k²(3π² - 1)·sin(kπx)·sin(kπy)"""
        x, y = p[..., 0], p[..., 1]
        kπ = bm.pi * self.k
        k = self.k
        return k**2 * (2 * bm.pi**2 - 1) * bm.sin(kπ * x) * bm.sin(kπ * y)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition: u = 0"""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary ∂Ω"""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x) < atol) | (bm.abs(x - 1.0) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y - 1.0) < atol)
        )
        return on_boundary
