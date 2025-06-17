from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinData1D:
    """
    1D Poisson problem:

        -u''(x) = f(x),  x in (0, 1)
         u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(πx)

    The corresponding source term is:

        f(x) = π²·sin(πx)

    Dirichlet boundary conditions are applied at both ends of the interval.
    """

    def geo_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return [0.0, 1.0]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x = p[..., 0]
        pi = bm.pi
        val = bm.sin(pi * x)
        return val

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Compute the gradient of the solution.
        Note: If the PDE model is one-dimensional, the tensor returned by 
        the gradient computation should match the shape of the input tensor p.
        """
        x = p
        pi = bm.pi
        val = pi * bm.cos(pi * x)
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        pi = bm.pi
        val = pi**2 * bm.sin(pi * x)
        return val

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p
        atol = 1e-12  # 绝对误差容限
    
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = ((bm.abs(x - 1.) < atol) | (bm.abs(x) < atol) )
        return on_boundary 

