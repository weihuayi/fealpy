from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from fealpy.backend import TensorLike
from fealpy.mesher.cylinder_mesher import CylinderMesher

class Exp0005(CylinderMesher):

    def __init__(self, options : dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.radius = 0.5
        self.height = 5.0
        self.lc = 0.2
        super().__init__(radius=self.radius, height=self.height, lc=self.lc)
        self.mesh = self.init_mesh()

    def get_dimension(self) -> int:
        return 3    
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-5
        on_boundary = (bm.abs(z) < atol)
        return on_boundary
    
    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(z - self.height) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-4
        on_boundary = (bm.abs(x**2 + y**2 - self.radius**2) < atol)
        return on_boundary
    
    @cartesian
    def is_pressure_boundary(self, p = None):
        if p is None:
            return 1
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return self.is_outlet_boundary(p)
        # return 0
    
    @cartesian
    def is_velocity_boundary(self, p):
        return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        # return None
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2] 
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    def pressure_integral_target(self) -> float:
        """Integral of the exact pressure over the domain."""
        return 0.0
    
    @cartesian
    def inlet_velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 2] = 1.0*(1 - (x**2 + y**2)/self.radius**2)
        return result
    
    @cartesian
    def outlet_velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape)
        u[..., 2] = 1.0*(1 - (x**2 + y**2)/self.radius**2)
        return u
    
    @cartesian
    def wall_velocity(self, p):
        u = bm.zeros(p.shape)
        return u
    
    @cartesian
    def velocity_dirichlet(self, p):
        result = bm.zeros(p.shape)
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        wall = self.wall_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_outlet = self.is_outlet_boundary(p)

        result[is_inlet] = inlet[is_inlet]
        result[is_wall] = wall[is_wall]
        # result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p):
        p = bm.zeros(p[..., 0].shape)
        return p
