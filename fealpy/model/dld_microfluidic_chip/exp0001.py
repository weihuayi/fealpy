from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class Exp0001():
    
    def __init__(self, **options):
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.radius = options.get("radius")
        self.centers = options.get("centers")
        self.inlet_boundary = options.get("inlet_boundary")
        self.outlet_boundary = options.get("outlet_boundary")
        self.wall_boundary = options.get("wall_boundary")

 
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
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        obstacle = self.is_obstacle_boundary(p)
        return inlet | wall | obstacle

    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return self.is_outlet_boundary(p)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]

        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        outlet = self.outlet_pressure(p)
        is_outlet = self.is_outlet_boundary(p)
        result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        bd = self.inlet_boundary
        return self.is_boundary(p, bd)

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        bd = self.outlet_boundary
        return self.is_boundary(p, bd)
    
    @cartesian
    def is_wall_boundary(self,p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        bd = self.wall_boundary
        return self.is_boundary(p, bd)
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = self.radius
        atol = 1e-12
        on_boundary = bm.zeros_like(x, dtype=bool)
        for center in self.centers:
            cx, cy = center
            on_boundary |= (x - cx)**2 + (y - cy)**2 < radius**2 + atol
        return on_boundary
    
    def is_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        atol = 1e-12
        v0 = p[:, None, :] - bd[None, 0::2, :] # (NN, NI, 2)
        v1 = p[:, None, :] - bd[None, 1::2, :] # (NN, NI, 2)

        cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
        dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
        cond = (bm.abs(cross) < atol) & (dot < atol)
        return bm.any(cond, axis=1)

