from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
import gmsh
from fealpy.geometry import DLDMicrofluidicChipModeler3D
from fealpy.mesher import DLDMicrofluidicChipMesher3D


class Cylinder3D():
    def __init__(self, options : dict = None):
        self.eps = 1e-10
        self.mu = 1e-3
        self.rho = 1.0
        self.options = options
        self.mesh = self.init_mesh()
        

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 3
    
    # def init_mesh(self):
    #     from fealpy.mesh import TetrahedronMesh
    #     import gmsh
    #     options = {
    #         "box": self.options.get("box", [0, 2.5, 0, 0.41, 0, 0.41]),
    #         "cyl_center": self.options.get("center", [0.5,0.2,0.0]),
    #         "cyl_axis": self.options.get("cyl_axis", [0.0, 0.0, 1.0]),
    #         "cyl_radius": self.options.get("radius", 0.05),
    #         "cyl_height":self.options.get("thickness", 0.10),
    #         "h": self.options.get("lc", 0.1)
    #         }
    #     mesh = TetrahedronMesh.from_box_minus_cylinder(**options)
    #     return mesh

    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 80*0.45*y*z*(0.41-y)*(0.41-z)/(0.41**4)
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape[0], dtype=bm.float64)
        return result
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        atol = 5e-3
        return bm.abs(p[..., 0]) < atol

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        atol = 0.01
        on_boundary = bm.abs((x - 2.5) <= atol)
        return on_boundary
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        y = p[..., 1]
        z = p[..., 2]
        atol = 5e-3
        on_boundary = (
            (bm.abs(y - 0.0) < atol) | (bm.abs(y - 0.41) < atol)|
            (bm.abs(z - 0.0) < atol) | (bm.abs(z - 0.41) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = 0.05
        atol = 0.01
        # 检查是否接近圆的边界
        on_boundary = (bm.abs((x - 0.5)**2 + (y - 0.2)**2 - radius**2)) < atol
        return on_boundary
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # out = self.is_outlet_boundary(p)
        # return ~out
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        # if p is None:
        #     return 1
        # is_out = self.is_outlet_boundary(p)
        # return is_out
        return 0
  
    @cartesian
    def velocity_dirichlet(self, p: TensorLike, t) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        outlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
    
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        # outlet = self.outlet_pressure(p)
        # is_outlet = self.is_outlet_boundary(p)

        # result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        # result[is_outlet] = outlet[is_outlet]
        # return result
        return self.outlet_pressure(p)
    
    @cartesian
    def velocity(self, p: TensorLike, t) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[0], dtype=p.dtype)
        return result