from typing import Sequence

from ....backend import backend_manager as bm
from ....decorator import cartesian
from ....backend import TensorLike
from ....mesher import BoxMesher2d

class Exp0002(BoxMesher2d):
    """
    2D Poisson problem with circular source and non-homogeneous boundary conditions:
    
        ∇²f(x,y) = -1 if √((x+2)² + y²) ≤ 1.5, else 0, (x,y) ∈ [-4,4]×[-4,4]
        f(-4,y) = 0
        f(4,y) = 1
        Natural boundary conditions on y=±4
    
    The scaling function that satisfies the boundary conditions is:
        B(x,y) = (x+4)/8 + (x²-4)y/128
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
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term"""
        x = p[..., 0]
        y = p[..., 1]
        
        # Calculate distance from (-2, 0)
        r = bm.sqrt((x + 2)**2 + y**2)
        
        # Source is -1 inside circle, 0 outside
        return -bm.where(r <= 1.5, 1.0, 0.0)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition (zero on all boundaries)
        
        Args:
            p: TensorLike, shape (..., 2), input points (x, y)
        
        Returns:
            TensorLike: Zero values at boundary points
        
        Raises:
            ValueError: If any point in 'p' is not on the boundary.
        """
        # 检查是否所有点都在边界上
        if not bm.all(self.is_dirichlet_boundary(p)):
            raise ValueError("All points in `p` must be on the boundary. "
                             "Use `is_dirichlet_boundary` to check boundary points.")
        x = p[..., 0]
        atol = 1e-12  # absolute tolerance
        
        # Only apply Dirichlet BC on x=-4 and x=4 boundaries
        on_left_boundary = bm.abs(x + 4) < atol
        on_right_boundary = bm.abs(x - 4) < atol
        
        # f(-4,y) = 0, f(4,y) = 1
        return bm.where(on_left_boundary, 0.0, bm.where(on_right_boundary, 1.0, 0.0))

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on Dirichlet boundary (x=±4)."""
        x = p[..., 0]
        atol = 1e-12  # absolute tolerance
        
        # Only x=-4 and x=4 have Dirichlet BC
        return (bm.abs(x + 4) < atol) | (bm.abs(x - 4) < atol)

    @cartesian
    def scaling_function(self, p: TensorLike) -> TensorLike:
        """Compute scaling function that satisfies the boundary conditions."""
        x = p[..., 0]
        y = p[..., 1]
        return (x + 4)/8 + (x**2 - 4)*y/128