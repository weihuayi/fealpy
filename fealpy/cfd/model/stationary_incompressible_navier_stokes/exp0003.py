from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0003(BoxMesher2d):
    
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 2.0
        self.mesh = self.init_mesh(nx=options.get('nx', 8), ny=options.get('ny', 8))
        super().__init__(box=self.box)
    
    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
 
    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = y*(1-y)
        result[..., 1] = 0.0
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[0], dtype=bm.float64)
        result[:] = 0.0
        return result
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # inlet = self.is_inlet_boundary(p)
        # wall = self.is_wall_boundary(p)
        # obstacle = self.is_obstacle_boundary(p)
        # return inlet | wall | obstacle
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result
        # return self.is_outlet_boundary(p)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        return self.inlet_velocity(p)
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        return self.outlet_pressure(p)
    
    @cartesian
    def is_inlet_boundary( p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 5e-3
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (bm.abs(x - 0.0) < atol)
        return bm.abs(p[..., 0]) < atol

    @cartesian
    def is_outlet_boundary( p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 5e-3
        on_boundary = (bm.abs(x - 1.0) < atol)
        return on_boundary
    @cartesian
    def is_wall_boundary(p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 5e-3
        on_boundary = (
            (bm.abs(y - 0.0) < atol) | (bm.abs(y - 1.0) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = 0.15
        atol = 5e-3
        # 检查是否接近圆的边界
        on_boundary = bm.zeros_like(x, dtype=bool)
        for center in bm.array([[0.3, 0.3], [0.3, 0.7], [0.7, 0.3], [0.7, 0.7]]):
        # for center in bm.array([[0.5, 0.5]]):
            cx, cy = center
            on_boundary |= bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
        return on_boundary

