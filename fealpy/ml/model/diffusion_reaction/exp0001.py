from typing import Sequence
from ....decorator import cartesian, variantmethod
from ....backend import backend_manager as bm
from ....backend import TensorLike
from ....mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    2D Poisson problem with reaction term and piecewise linear boundary conditions:
    
        ∇²f(x,y) + (1/5)f(x,y) = e^{-(x²+y²)²}, (x,y) ∈ [-3,3]^2
        
    Boundary conditions:
        f(-3,-3)=0.1, f(-3,3)=0.2, f(3,3)=0.3, f(3,-3)=0.4
        Linear interpolation between these points:
        f(x,-3) = 0.05x + 0.25
        f(x,3) = (0.05/3)x + 0.25
        f(-3,y) = (0.05/3)y + 0.15
        f(3,y) = -(0.05/3)y + 0.35
    
    Scaling function that satisfies the boundary conditions:
        B(x,y) = x/30 - xy/180 + 0.25
    """
    def __init__(self):
        self.box = [-3.0, 3.0, -3.0, 3.0]  # [xmin, xmax, ymin, ymax]
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        """Diffusion coefficient (identity matrix)"""
        val = bm.array([[1.0, 0.0], [0.0, 1.0]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)
    
    @cartesian
    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        """Inverse diffusion coefficient (identity matrix)"""
        val = bm.array([[1.0, 0.0], [0.0, 1.0]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def reaction_coef(self, p: TensorLike) -> TensorLike:
        """Reaction coefficient (1/5)"""
        val = bm.array([0.2])  # 1/5 = 0.2
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Source term: e^{-(x²+y²)²}"""
        x = p[..., 0]
        y = p[..., 1]
        r_squared = x**2 + y**2
        return bm.exp(-r_squared**2)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition with piecewise linear interpolation"""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        
        # Check which boundary the point is on
        on_left = bm.abs(x + 3) < atol
        on_right = bm.abs(x - 3) < atol
        on_bottom = bm.abs(y + 3) < atol
        on_top = bm.abs(y - 3) < atol
        
        # Corner points (exact values)
        val = bm.where((on_left & on_bottom), 0.1,
                       bm.where((on_left & on_top), 0.2,
                                bm.where((on_right & on_top), 0.3,
                                         bm.where((on_right & on_bottom), 0.4, 0.0))))
        
        # Linear interpolation for edges
        val = bm.where(on_bottom & (val == 0), 0.05*x + 0.25, val)
        val = bm.where(on_top    & (val == 0), (0.05/3)*x + 0.25, val)
        val = bm.where(on_left   & (val == 0), (0.05/3)*y + 0.15, val)
        val = bm.where(on_right  & (val == 0), -(0.05/3)*y + 0.35, val) 
        return val

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12  # absolute tolerance
        
        return ((bm.abs(x + 3) < atol) | (bm.abs(x - 3) < atol) |
            (bm.abs(y + 3) < atol) | (bm.abs(y - 3) < atol))

    @cartesian
    def scaling_function(self, p: TensorLike) -> TensorLike:
        """Scaling function that satisfies the boundary conditions"""
        x = p[..., 0]
        y = p[..., 1]
        return x/30 - (x*y)/180 + 0.25
