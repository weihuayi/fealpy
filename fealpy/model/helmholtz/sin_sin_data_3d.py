from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..box_domain_mesher import BoxDomainMesher3d
from ...decorator import cartesian

class SinSinData3D(BoxDomainMesher3d):
    """
    3D Helmholtz problem with homogeneous Dirichlet boundary condition:
    
        -Δu(x, y, z) - k^2·u(x, y, z) = f(x, y, z),  (x, y, z) ∈ (0, 1)^3
                                u(x, y, z) = 0,     on ∂Ω

    with the exact solution:

        u(x, y, z) = sin(kx)·sin(ky)·sin(kz)

    The corresponding source term is:

        f(x, y, z) = 2k^2·sin(kx)·sin(ky)·sin(kz)

    Parameter:
        k : wave number (scalar)

    Source:
        https://link.springer.com/book/10.1007/b98828
    """

    def set(self, k: float=1.0):
        self.k = k

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u(x, y, z) = sin(kx)·sin(ky)·sin(kz)"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        k = self.k
        return bm.sin(k * x) * bm.sin(k * y) * bm.sin(k * z)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of the exact solution"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        k = self.k
        return bm.stack((
            k * bm.cos(k * x) * bm.sin(k * y) * bm.sin(k * z),
            k * bm.sin(k * x) * bm.cos(k * y) * bm.sin(k * z),
            k * bm.sin(k * x) * bm.sin(k * y) * bm.cos(k * z)
        ), axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term f = 2k^2·sin(kx)·sin(ky)·sin(kz)"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        k = self.k
        return 2 * k**2 * bm.sin(k * x) * bm.sin(k * y) * bm.sin(k * z)

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
