from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.geometry import DLDModeler
from fealpy.mesher import DLDMesher

import sympy as sp

class Exp0002(DLDMesher):
    def __init__(self, options: dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1.0e-3
        self.rho = 1.0e3
        self.box = options.get('box', [0.0, 0.75, 0.0, 0.41])
        self.radius = options.get('radius', 0.029)
        super().__init__(options)

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.options['box']
    
    def init_mesh(self):
        import gmsh
        import ipdb
        gmsh.initialize()
        
        modeler = DLDModeler(options=self.options)
        modeler.build(gmsh)

        self.generate(modeler, gmsh)
        ipdb.set_trace()
        gmsh.finalize()
        
        return self.mesh
    
    @cartesian
    def velocity_dirichlet(self, p:TensorLike) -> TensorLike:
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        well = self.wall_velocity(p)
        obstacle = self.obstacle_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_obstacle = self.is_obstacle_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        result[is_wall] = well[is_wall]
        result[is_obstacle] = obstacle[is_obstacle]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        outlet = self.outlet_pressure(p)
        well = self.wall_pressure(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)

        result = bm.zeros(p.shape[0], dtype=p.dtype)
        result[is_wall] = well[is_wall]
        result[is_outlet] = outlet[is_outlet]
        return result

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.03
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.03
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[0], dtype=p.dtype)
        result[:] = 0.0
        return result
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on wall."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.0
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def wall_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure on wall."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[0], dtype=p.dtype)
        result[:] = 0.0
        return result
    
    @cartesian
    def obstacle_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on obstacle."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.array(0.0)
        return result
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.0
        result[..., 1] = 0.0
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def is_pressure_boundary(self, p):
        # is_out = self.is_outlet_boundary(p)
        # return is_out
        return 0
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (bm.abs(x - self.box[0]) < atol)
        return on_boundary

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (bm.abs(x - self.box[1]) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(y - self.box[2]) < atol) | (bm.abs(y - self.box[3]) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = self.radius
        atol = 1e-12
        # 检查是否接近圆的边界
        on_boundary = bm.zeros_like(x, dtype=bool)
        for center in self.centers:
            cx, cy = center
            on_boundary |= bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
        return on_boundary
        