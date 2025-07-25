from typing import Sequence

from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0009(BoxMesher2d):
    """
    2D Poisson problem with circular source and non-homogeneous boundary conditions:

        -Δu(x,y) = f(x,y),   (x,y) ∈ [-4,4]^2
        u(x,y) = g(x, y), (x,y) ∈ ∂Ω
    
    with the exact solution:

        Unknown.

    The corresponding source term is:

        f(x,y) = 1 if √((x+2)² + y²) ≤ 1.5, else 0
   
    The boundary conditions are:
    
        u(-4,y) = 0,  u(4,y) = 1,
        the remaining two edges with quadratic interpolated values along the edges.

    Reference:
        https://doi.org/10.1016/j.neucom.2024.128936
    """
    def __init__(self):
        self.box = [-4.0, 4.0, -4.0, 4.0]  # [xmin, xmax, ymin, ymax]
        super().__init__(box=self.box)

    def get_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike):
        """Exact solution is unknown, return NotImplementedError."""
        raise NotImplementedError("The exact solution is unknown.")

    @cartesian
    def gradient(self, p: TensorLike):
        """Gradient is unknown, return NotImplementedError."""
        raise NotImplementedError("The exact gradient is unknown.")

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term"""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt((x + 2)**2 + y**2)
        return bm.where(r <= 1.5, 1.0, 0.0)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition."""
        # 检查是否所有点都在边界上
        # if not bm.all(self.is_dirichlet_boundary(p)):
        #     raise ValueError("All points in `p` must be on the boundary. "
        #                      "Use `is_dirichlet_boundary` to check boundary points.")
        x = p[..., 0]
        atol = 1e-12  # absolute tolerance
        
        # Only apply Dirichlet BC on x=-4 and x=4 boundaries
        on_left_boundary = bm.abs(x + 4) < atol
        on_right_boundary = bm.abs(x - 4) < atol
        
        # u(-4,y) = 0, u(4,y) = 1
        val = bm.where(on_left_boundary, 0.0, bm.where(on_right_boundary, 1.0, 0.0))
        return val.to(p.dtype)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on Dirichlet boundary (x=±4)."""
        x = p[..., 0]
        atol = 1e-5  # absolute tolerance
        
        # Only x=-4 and x=4 have Dirichlet BC
        return (bm.abs(x + 4) < atol) | (bm.abs(x - 4) < atol)

    @cartesian
    def scaling_function(self, p: TensorLike) -> TensorLike:
        """Compute scaling function that satisfies the boundary conditions."""
        x, y = p[..., 0], p[..., 1]
        return (x + 4)/8 + (x**2 - 4)*y/128
    
    def identify_boundary_edge(self) -> tuple:
        """Determine which edges of a 2D rectangular domain are boundary edges for the PDE.
    
        Identifies the boundary edges of the computational domain where boundary conditions
        are applied. The function returns a tuple indicating which edges (left, right, bottom, top)
        are designated as boundary edges for the partial differential equation.

        Returns
            boundary_edges : tuple[int]
                Tuple containing the indices of boundary edges. Possible values:
                0: Left boundary (x = x_min)
                1: Right boundary (x = x_max)
                2: Bottom boundary (y = y_min)
                3: Top boundary (y = y_max)
                Example: (0, 1) indicates both left and right boundaries are PDE boundaries.
        """
        return (0, 1)