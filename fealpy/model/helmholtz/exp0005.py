from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..mesher import BoxMesher3d
from ...decorator import cartesian

class Exp0005(BoxMesher3d):
    """
    3D Helmholtz problem with homogeneous Dirichlet boundary condition:

        -Δu(x, y, z) - k^2·u(x, y, z) = f(x, y, z),  in Ω = (0, 1)^3
                               u(x, y, z) = 0,      on ∂Ω

    with the exact solution:

        u(x, y, z) = sin(kπx)·sin(kπy)·sin(kπz)

    The corresponding source term is:

        f(x, y, z) = k^2·(3π^2 - 1)·sin(kπx)·sin(kπy)·sin(kπz)

    Parameter:
        k : wave number (scalar)

    Reference:
        https://link.springer.com/book/10.1007/b98828
    """

    def __init__(self, options: dict = {}):  
        self.box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)
        self.k = options.get('k', 1.0)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return self.box
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u(x, y, z) = sin(kπx)·sin(kπy)·sin(kπz)"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        kπ = bm.pi * self.k
        return bm.sin(kπ * x) * bm.sin(kπ * y) * bm.sin(kπ * z)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of the exact solution"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        kπ = bm.pi * self.k
        return bm.stack((
            kπ * bm.cos(kπ * x) * bm.sin(kπ * y) * bm.sin(kπ * z),
            kπ * bm.sin(kπ * x) * bm.cos(kπ * y) * bm.sin(kπ * z),
            kπ * bm.sin(kπ * x) * bm.sin(kπ * y) * bm.cos(kπ * z),
        ), axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term f = k^2·(3π^2 - 1)·sin(kπx)·sin(kπy)·sin(kπz)"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        kπ = bm.pi * self.k
        k = self.k
        return k**2 * (3 * bm.pi**2 - 1) * bm.sin(kπ * x) * bm.sin(kπ * y) * bm.sin(kπ * z)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition: u = 0"""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary ∂Ω"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x) < atol) | (bm.abs(x - 1.0) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y - 1.0) < atol) |
            (bm.abs(z) < atol) | (bm.abs(z - 1.0) < atol)
        )
        return on_boundary
