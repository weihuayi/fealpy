from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike


class TriharmonicData2D():
    """
    2D triharmonic problem:

        Δ³u = f,  with u = ∂u/∂n = ∂^2u/∂n^2  = 0 on ∂Ω

    Exact solution: u(x, y) = sin(2πx)·cos(2πy)
    """
    def geo_dimension(self):
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self):
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0., 1., 0., 1.]

    def init_mesh(self, nx=10, ny=10):
        from ...mesh import TriangleMesh
        return TriangleMesh.from_box(self.domain(), nx=nx, ny=ny)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        pi =  bm.pi
        return bm.sin(2*pi * x) * bm.sin(2*pi * y)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        ux = 2*pi*sin(2*pi*y)*cos(2*pi*x)
        uy = 2*pi*sin(2*pi*x)*cos(2*pi*y)
        return bm.stack((ux, uy), axis=-1)

    @cartesian
    def hessian(self, p: TensorLike) -> TensorLike:
        """Compute hessian of solution."""
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        pi =  bm.pi
        uxx = -4*pi**2*sin(2*pi*x)*sin(2*pi*y)
        uxy = 4*pi**2*cos(2*pi*x)*cos(2*pi*y)
        uyy = -4*pi**2*sin(2*pi*x)*sin(2*pi*y)
        return bm.stack((uxx, uxy, uyy), axis=-1)

    @cartesian
    def grad_3(self, p: TensorLike) -> TensorLike:
        """Compute third order gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        pi =  bm.pi
        uxxx = -8*pi**3*sin(2*pi*y)*cos(2*pi*x)
        uxxy = -8*pi**3*sin(2*pi*x)*cos(2*pi*y)
        uxyy = -8*pi**3*sin(2*pi*y)*cos(2*pi*x)
        uyyy = -8*pi**3*sin(2*pi*x)*cos(2*pi*y)
        return bm.stack((uxxx, uxxy, uxyy, uyyy), axis=-1)

    @cartesian
    def grad_4(self, p: TensorLike) -> TensorLike:
        """Compute fourth order gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        pi =  bm.pi
        uxxxx = 16*pi**4*sin(2*pi*x)*sin(2*pi*y)
        uyxxx = -16*pi**4*cos(2*pi*x)*cos(2*pi*y)
        uyyxx = 16*pi**4*sin(2*pi*x)*sin(2*pi*y)
        uyyyx = -16*pi**4*cos(2*pi*x)*cos(2*pi*y)
        uyyyy = 16*pi**4*sin(2*pi*x)*sin(2*pi*y)
        return bm.stack((uxxxx, uyxxx, uyyxx, uyyyx, uyyyy), axis=-1)


    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x, y = p[..., 0], p[..., 1]
        pi =  bm.pi
        sin = bm.sin
        cos = bm.cos
        L3u = 512*pi**6*sin(2*pi*x)*sin(2*pi*y)
        return L3u

    def get_flist(self):
        """Return the list of functions."""
        return [self.solution, self.gradient, self.hessian, self.grad_3, self.grad_4]

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)


