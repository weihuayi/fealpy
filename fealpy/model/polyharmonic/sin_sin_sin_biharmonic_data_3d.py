from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class SinSinSinBiharmonicData3D():
    """
    3D biharmonic problem:

        Δ²u = f  in Ω = (0,1)^3,
        u = ∂u/∂n = 0  on ∂Ω

    with exact solution:
        u(x,y,z) = sin(5x)·sin(5y)·sin(5z)
    """

    def geo_dimension(self):
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self):
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return [0., 1., 0., 1., 0., 1.]

    def init_mesh(self, nx=1, ny=1, nz=1):
        """Initialize a mesh for the domain."""
        from ...mesh import TetrahedronMesh
        return TetrahedronMesh.from_box(self.domain(), nx=nx, ny=ny, nz=nz)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Exact solution of the biharmonic problem."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return bm.sin(5 * x) * bm.sin(5 * y) * bm.sin(5 * z)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Gradient of the exact solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        ux = 5 * bm.cos(5 * x) * bm.sin(5 * y) * bm.sin(5 * z)
        uy = 5 * bm.sin(5 * x) * bm.cos(5 * y) * bm.sin(5 * z)
        uz = 5 * bm.sin(5 * x) * bm.sin(5 * y) * bm.cos(5 * z)
        return bm.stack((ux, uy, uz), axis=-1)

    @cartesian
    def hessian(self, p: TensorLike) -> TensorLike:
        """Hessian of the exact solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        uxx = -25 * bm.sin(5 * x) * bm.sin(5 * y) * bm.sin(5 * z)
        uxy = 25 * bm.cos(5 * x) * bm.cos(5 * y) * bm.sin(5 * z)
        uxz = 25 * bm.cos(5 * x) * bm.sin(5 * y) * bm.cos(5 * z)
        uyy = -25 * bm.sin(5 * x) * bm.sin(5 * y) * bm.sin(5 * z)
        uyz = 25 * bm.sin(5 * x) * bm.cos(5 * y) * bm.cos(5 * z)
        uzz = -25 * bm.sin(5 * x) * bm.sin(5 * y) * bm.sin(5 * z)
        return bm.stack((uxx,uxy,uxz,uyy,uyz,uzz), axis=-1)

    @cartesian
    def grad_3(self, p: TensorLike) -> TensorLike:
        """Third order gradient of the exact solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        sin = bm.sin
        cos = bm.cos

        uxxx = -125 * sin(5*y) * sin(5*z) * cos(5*x)
        uxxy = -125 * sin(5*x) * sin(5*z) * cos(5*y)  # = uyxx
        uxxz = -125 * sin(5*x) * sin(5*y) * cos(5*z)
        uxyy = -125 * sin(5*y) * sin(5*z) * cos(5*x)  # = uyyx
        uxyz =  125 * cos(5*x) * cos(5*y) * cos(5*z)
        uxzz = -125 * sin(5*y) * sin(5*z) * cos(5*x)
        uyyy = -125 * sin(5*x) * sin(5*z) * cos(5*y)
        uyyz = -125 * sin(5*x) * sin(5*y) * cos(5*z)
        uyzz = -125 * sin(5*x) * sin(5*z) * cos(5*y)
        uzzz = -125 * sin(5*x) * sin(5*y) * cos(5*z)

        val = bm.stack([ uxxx, uxxy, uxxz, uxyy, uxyz, uxzz, uyyy, uyyz, uyzz, uzzz ], axis=-1)
        return val

    @cartesian
    def grad_4(self, p: TensorLike) -> TensorLike:
        """Fourth order gradient of the exact solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        sin = bm.sin
        cos = bm.cos

        uxxxx =  625 * sin(5*x) * sin(5*y) * sin(5*z)
        uyxxx = -625 * sin(5*z) * cos(5*x) * cos(5*y)
        uxxxz = -625 * sin(5*y) * cos(5*x) * cos(5*z)
        uyyxx =  625 * sin(5*x) * sin(5*y) * sin(5*z)
        uxxyz = -625 * sin(5*x) * cos(5*y) * cos(5*z)
        uxxzz =  625 * sin(5*x) * sin(5*y) * sin(5*z)
        uyyyx = -625 * sin(5*z) * cos(5*x) * cos(5*y)
        uxyyz = -625 * sin(5*y) * cos(5*x) * cos(5*z)
        uxyzz = -625 * sin(5*z) * cos(5*x) * cos(5*y)
        uxzzz = -625 * sin(5*y) * cos(5*x) * cos(5*z)
        uyyyy =  625 * sin(5*x) * sin(5*y) * sin(5*z)
        uyyyz = -625 * sin(5*x) * cos(5*y) * cos(5*z)
        uyyzz =  625 * sin(5*x) * sin(5*y) * sin(5*z)
        uyzzz = -625 * sin(5*x) * cos(5*y) * cos(5*z)
        uzzzz =  625 * sin(5*x) * sin(5*y) * sin(5*z)

        val = bm.stack([ uxxxx, uyxxx, uxxxz, uyyxx, uxxyz, uxxzz, uyyyx, uxyyz, uxyzz, uxzzz, uyyyy, uyyyz, uyyzz, uyzzz, uzzzz ], axis=-1)
        return val
            

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Source term for the biharmonic equation."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        sin = bm.sin
        cos = bm.cos
        return 5625*sin(5*x)*sin(5*y)*sin(5*z)

    def get_flist(self) -> Sequence[TensorLike]:
        """Return the list of source terms"""
        return [self.solution, self.gradient, self.hessian, self.grad_3, self.grad_4]

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

