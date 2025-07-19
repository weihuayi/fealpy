from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0005(BoxMesher2d):
    """
    2D Poisson problem with interior layer on a square domain:
    
        -Δu(x, y) = f(x, y),  (x, y) ∈ (-1, 1) x (-1, 1)
         u(x, y) = g(x, y),    on ∂Ω

    with the exact solution:

        u(x, y) = 1 / (1 + exp^(-100·(r - 0.75)))

    The corresponding source term is:

        f(x, y) = -100·u·(1 - u)·[100·(1 - 2u) + 1/r]
    """
    def __init__(self):
        self.box = [-1.0, 1.0, -1.0, 1.0] 
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    @variantmethod('tri')
    def init_mesh(self, nx=16, ny=16):
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
        """Compute exact solution"""
        r = bm.linalg.norm(p, axis=-1)
        val = 1 / (1 + bm.exp(-100 * (r - 0.75)))
        return val 

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        u = self.solution(p)
        m = 100 * u * (1 - u) / bm.linalg.norm(p, axis=-1)
        val = m[..., None] * p
        return val
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        r = bm.linalg.norm(p, axis=-1)
        u = self.solution(p)
        val = -100 * u * (1 - u) * (100 * (1 - 2 * u) + 1 / r)
        return val

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""        
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12 
        on_boundary = (
            (bm.abs(x - 0.) < atol) | (bm.abs(x - 1.) < atol) |
            (bm.abs(y - 0.) < atol) | (bm.abs(y - 1.) < atol)
        )
        return on_boundary