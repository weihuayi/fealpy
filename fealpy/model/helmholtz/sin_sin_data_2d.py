from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinSinData2D():
    """
    2D Helmholtz problem with homogeneous Dirichlet boundary condition:
    
        -Δu(x, y) - k^2·u(x, y) = f(x, y),  (x, y) ∈ (0, 1) x (0, 1)
                           u(x, y) = 0,    on ∂Ω

    with the exact solution:

        u(x, y) = sin(kx)·sin(ky)

    The corresponding source term is:

        f(x, y) = k^2·sin(kx)·sin(ky)

    Parameter:
        k : wave number (scalar)
    """

    def __init__(self, k: float):
        self.k = k

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]
    
    @variantmethod('tri')
    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        d = self.domain()
        mesh = TriangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh 
     
    @init_mesh.register('quad')
    def init_mesh(self, nx=10, ny=10):
        from ...mesh import QuadrangleMesh
        d = self.domain()
        mesh = QuadrangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u(x, y) = sin(kx)·sin(ky)"""
        x, y = p[..., 0], p[..., 1]
        k = self.k
        return bm.sin(k * x) * bm.sin(k * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        k = self.k
        val = bm.stack((
            k * bm.cos(k * x) * bm.sin(k * y),
            k * bm.sin(k * x) * bm.cos(k * y)
        ), axis=-1)
        return val  # shape == (N, 2)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term f(x, y) = k²·sin(kx)·sin(ky)"""
        x, y = p[..., 0], p[..., 1]
        k = self.k
        return k**2 * bm.sin(k * x) * bm.sin(k * y)

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
