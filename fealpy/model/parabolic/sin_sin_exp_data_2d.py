from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinSinExpData2D:
    """
    2D parabolic problem:

        ∂u/∂t - (∂²u/∂x² + ∂²u/∂y²) = f   (x, y) ∈ (0, 1) x (0, 1), t > 0

        u(x, 0, t) = u(x, 1, t) = 0,         x ∈ (0, 1), t > 0
        u(0, y, t) = u(1, y, t) = 0,         y ∈ (0, 1), t > 0
        u(x, y, 0) = sin(4πx)·sin(4πy),     (x, y) ∈ (0, 1) x (0, 1)

    Exact solution:

        u(x, y, t) = sin(4πx)·sin(4πy)·exp(-20t)
        f(x, y, t) = -20·exp(-20t)·sin(4πx)·sin(4πy)
                    + 32π²·exp(-20t)·sin(4πx)·sin(4πy),  

    This example imposes homogeneous Dirichlet boundary conditions on all four edges.
    It is useful for verifying time-dependent solvers in 2D.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]  

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.sin(4 * bm.pi * x) * bm.sin(4 * bm.pi * y)

    @cartesian
    def solution(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x, y = p[..., 0], p[..., 1]
        return bm.sin(4 * bm.pi * x) * bm.sin(4 * bm.pi * y) * bm.exp(-20 * t)

    @cartesian
    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        x, y = p[..., 0], p[..., 1]
        factor = bm.exp(-20 * t)
        gx = 4 * bm.pi * bm.cos(4 * bm.pi * x) * bm.sin(4 * bm.pi * y) * factor
        gy = 4 * bm.pi * bm.sin(4 * bm.pi * x) * bm.cos(4 * bm.pi * y) * factor
        return bm.stack([gx, gy], axis=-1)  # shape (..., 2)

    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x, y = p[..., 0], p[..., 1]
        factor = bm.exp(-20 * t) * bm.sin(4 * bm.pi * x) * bm.sin(4 * bm.pi * y)
        return (-20 + 32 * bm.pi ** 2) * factor

    @cartesian
    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        """Check if point is on boundary."""
        return self.solution(p, t)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) | \
               (bm.abs(y - 0.0) < 1e-12) | (bm.abs(y - 1.0) < 1e-12)


