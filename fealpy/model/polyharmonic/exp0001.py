from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    2D biharmonic problem:

        Δ²u = f  in Ω = (-1,1)x(-1,1),
        u = ∂u/∂n = 0  on ∂Ω

    with exact solution:
        u(x,y) = (sin(2πx)·sin(2πy))²
    """
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)

    def geo_dimension(self):
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self):
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        pi =  bm.pi
        return bm.sin(2*pi * x)**2 * bm.sin(2*pi * y)**2

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        ux = 4*pi*sin(2*pi*x)*sin(2*pi*y)**2*cos(2*pi*x)
        uy = 4*pi*sin(2*pi*x)**2*sin(2*pi*y)*cos(2*pi*y)
        return bm.stack((ux, uy), axis=-1)

    @cartesian
    def hessian(self, p: TensorLike) -> TensorLike:
        """Compute hessian of solution."""
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        pi =  bm.pi
        uxx = 8*pi**2*(-sin(2*pi*x)**2 + cos(2*pi*x)**2)*sin(2*pi*y)**2
        uxy = 16*pi**2*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*x)*cos(2*pi*y)
        uyy = 8*pi**2*(-sin(2*pi*y)**2 + cos(2*pi*y)**2)*sin(2*pi*x)**2
        return bm.stack((uxx, uxy, uyy), axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x, y = p[..., 0], p[..., 1]
        pi =  bm.pi
        sin = bm.sin
        cos = bm.cos
        L2u = 128*pi**4*(3*sin(2*pi*x)**2*sin(2*pi*y)**2 - 2*sin(2*pi*x)**2*cos(2*pi*y)**2 - 2*sin(2*pi*y)**2*cos(2*pi*x)**2 + cos(2*pi*x)**2*cos(2*pi*y)**2)
        return L2u

    def get_flist(self) -> Sequence[TensorLike]:
        """Return the list of source terms"""
        return [self.solution, self.gradient, self.hessian]


    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)


