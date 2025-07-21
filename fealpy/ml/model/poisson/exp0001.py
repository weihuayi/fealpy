from typing import Sequence

from ....backend import backend_manager as bm
from ....decorator import cartesian
from ....backend import TensorLike
from ....mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    2D Poisson problem:
    
        ∇²f(x,y) = e^{-((x+2)²/2 + y²/2)} - 1/2 e^{-((x-2)²/2 + y²/2)}, (x,y) ∈ [-5,5]^2
        f(x,y) = 0, (x,y) ∈ ∂Ω
    
    The domain is a square with Dirichlet boundary conditions applied on all boundaries.
    """
    def __init__(self):
        self.box= [-5.0, 5.0, -5.0, 5.0]  # [xmin, xmax, ymin, ymax]
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
        
        # First Gaussian term
        term1 = bm.exp(-((x + 2)**2 + y**2) / 2)
        # Second Gaussian term
        term2 = 0.5 * bm.exp(-((x - 2)**2 + y**2) / 2)
        
        return term1 - term2

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition (zero on all boundaries)"""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12  # absolute tolerance
        
        # Check if close to any of the boundaries
        on_boundary = ((bm.abs(x + 5) < atol) | (bm.abs(x - 5) < atol) |
                       (bm.abs(y + 5) < atol) | (bm.abs(y - 5) < atol))
        return on_boundary
    
    def scaling_function(self, p: TensorLike) -> TensorLike:
        """Compute scaling function that satisfies the boundary conditions."""
        return bm.ones_like(p[..., 0])