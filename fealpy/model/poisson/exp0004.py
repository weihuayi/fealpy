from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike


class Exp0004:
    """
    L-shaped domain corner singularity problem (2D Poisson):

        -Δu(x, y) = f(x, y),  (x, y) ∈ (-1, 1)^2 / [0, 1) x (-1, 0]
         u(x, y) = g(x, y),    on ∂Ω

    with the exact solution:

        u(x, y) = r**(2/3) * sin(2/3*theta)

    The corresponding source term is:

        f(x, y) = 0
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2
    
    @variantmethod('tri')
    def init_mesh(self, n=3):
        from ...mesh import TriangleMesh
        node = bm.array([
            [-1, -1],
            [ 0, -1],
            [-1,  0],
            [ 0,  0],
            [ 1,  0],
            [-1,  1],
            [ 0,  1],
            [ 1,  1]], dtype=bm.float64)
        cell = bm.array([
            [1, 3, 0],
            [2, 0, 3],
            [3, 6, 2],
            [5, 2, 6],
            [4, 7, 3],
            [6, 3, 7]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    @init_mesh.register('quad')
    def init_mesh(self, n):
        from ...mesh import QuadrangleMesh
        node = bm.array([
            [-1, -1],
            [ 0, -1],
            [-1,  0],
            [ 0,  0],
            [ 1,  0],
            [-1,  1],
            [ 0,  1],
            [ 1,  1]], dtype=bm.float64)
        cell = bm.array([
            [0, 1, 3, 2],
            [3, 4, 7, 6],
            [3, 6, 5, 2]], dtype=bm.int32)
        mesh = QuadrangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        r = bm.linalg.norm(p, axis=-1)
        theta = bm.arctan2(y, x) % (2 * bm.pi)
        val = r**(2.0/3) * bm.sin(2.0/3 * theta)
        return val 

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        r = bm.linalg.norm(p, axis=-1)
        theta = bm.arctan2(y, x) % (2 * bm.pi)
        dx = -2/3 * r**(-1.0/3) * bm.sin(1.0/3 * theta)
        dy =  2/3 * r**(-1.0/3) * bm.cos(1.0/3 * theta)
        return bm.stack([dx, dy], axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        val = bm.zeros(len(p), dtype=bm.float64)
        return val

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""        
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # 绝对误差容限
    
        # 检查是否接近 x=±1 或 y=±1 或 x=0&<0 或 y=0&x>0
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x + 1.) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y + 1.) < atol) |
            ((bm.abs(x - 0.) < atol) & ( y < 0)) |
            ((bm.abs(y - 0.) < atol) & ( x > 0))
        )
        return on_boundary