from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from typing import Union, Callable, Dict
from fealpy.mesher.box_mesher import BoxMesher3d 

class Exp0004(BoxMesher3d):

    def __init__(self, options : dict = {}):
        self.options = options
        self.eps = 1e-10
        self.box = [0, 1, 0, 1, 0, 1]
        self.mu = 1.0
        self.rho = 1.0
        super().__init__(options.get("box", [0, 1, 0, 1, 0, 1]))
        self.mesh = self.init_mesh(nx = options.get("nx", 2), ny = options.get("ny", 2), nz = options.get("nz", 2))

    def get_dimension(self) -> int:
        return 3    

    def domain(self):
        return self.box
    
    @cartesian
    def is_pressure_boundary(self, p):
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2] 
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = x**3*y**2*z**2*(x - 1)**3*(16.0*x - 8.0)*(y - 1)**2*(z - 1)**2*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1))**2 + 4.0*x**3*y**2*z**2*(x - 1)**3*(y - 1)**2*(z - 1)**2*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1))*(2*y*z*(y - 1)*(z - 1) + y*z*(y - 1)*(2*z - 1) + y*(y - 1)*(z - 1)*(2*z - 1) - 2*z**2*(2*y - 1)*(z - 1) - 2*z*(2*y - 1)*(z - 1)**2) + 4.0*x**3*y**2*z**2*(x - 1)**3*(y - 1)**2*(z - 1)**2*(x*(x - 1)*(2*z - 1) - z*(2*x - 1)*(z - 1))*(-2*y**2*(y - 1)*(2*z - 1) + 2*y*z*(y - 1)*(z - 1) + y*z*(2*y - 1)*(z - 1) - 2*y*(y - 1)**2*(2*z - 1) + z*(y - 1)*(2*y - 1)*(z - 1)) + 4.0*x**2*y*z*(y - 1)*(z - 1)*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1)) + x**2*y*(x - 1)**2*(y - 1)*(8.0*y*z*(y - 1) + 8.0*y*(y - 1)*(z - 1) + 4.0*y*(y - 1)*(2*z - 1) + z**2*(4.0 - 8.0*y) - 16.0*z*(2*y - 1)*(z - 1) + (4.0 - 8.0*y)*(z - 1)**2) + x**2*z*(x - 1)**2*(z - 1)*(y**2*(8.0*z - 4.0) - 8.0*y*z*(z - 1) + 16.0*y*(y - 1)*(2*z - 1) + 4.0*z*(1 - 2*y)*(z - 1) + 8.0*z*(1 - y)*(z - 1) + (y - 1)**2*(8.0*z - 4.0)) + 16.0*x*y*z*(x - 1)*(y - 1)*(z - 1)*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1)) + 4.0*y*z*(x - 1)**2*(y - 1)*(z - 1)*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1)) + 2*(2*y - 1)*(2*z - 1)
        result[..., 1] = x**2*y**3*z**2*(x - 1)**2*(y - 1)**3*(16.0*y - 8.0)*(z - 1)**2*(x*(x - 1)*(2*z - 1) - z*(2*x - 1)*(z - 1))**2 - 4.0*x**2*y**3*z**2*(x - 1)**2*(y - 1)**3*(z - 1)**2*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1))*(2*x*z*(x - 1)*(z - 1) + x*z*(x - 1)*(2*z - 1) + x*(x - 1)*(z - 1)*(2*z - 1) - 2*z**2*(2*x - 1)*(z - 1) - 2*z*(2*x - 1)*(z - 1)**2) + 4.0*x**2*y**3*z**2*(x - 1)**2*(y - 1)**3*(z - 1)**2*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1))*(-2*x**2*(x - 1)*(2*z - 1) + 2*x*z*(x - 1)*(z - 1) + x*z*(2*x - 1)*(z - 1) - 2*x*(x - 1)**2*(2*z - 1) + z*(x - 1)*(2*x - 1)*(z - 1)) + 4.0*x*y**2*z*(x - 1)*(z - 1)*(-x*(x - 1)*(2*z - 1) + z*(2*x - 1)*(z - 1)) + x*y**2*(x - 1)*(y - 1)**2*(-8.0*x*z*(x - 1) - 8.0*x*(x - 1)*(z - 1) - 4.0*x*(x - 1)*(2*z - 1) + z**2*(8.0*x - 4.0) + 16.0*z*(2*x - 1)*(z - 1) + (8.0*x - 4.0)*(z - 1)**2) - 16.0*x*y*z*(x - 1)*(y - 1)*(z - 1)*(x*(x - 1)*(2*z - 1) - z*(2*x - 1)*(z - 1)) + 4.0*x*z*(x - 1)*(y - 1)**2*(z - 1)*(-x*(x - 1)*(2*z - 1) + z*(2*x - 1)*(z - 1)) + y**2*z*(y - 1)**2*(z - 1)*(x**2*(4.0 - 8.0*z) + 8.0*x*z*(z - 1) - 16.0*x*(x - 1)*(2*z - 1) - 4.0*z*(1 - 2*x)*(z - 1) - 8.0*z*(1 - x)*(z - 1) + (4.0 - 8.0*z)*(x - 1)**2) + 2*(2*x - 1)*(2*z - 1)
        result[..., 2] = x**2*y**2*z**3*(x - 1)**2*(y - 1)**2*(z - 1)**3*(16.0*z - 8.0)*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1))**2 - 4.0*x**2*y**2*z**3*(x - 1)**2*(y - 1)**2*(z - 1)**3*(x*(x - 1)*(2*z - 1) - z*(2*x - 1)*(z - 1))*(2*x*y*(x - 1)*(y - 1) + x*y*(x - 1)*(2*y - 1) + x*(x - 1)*(y - 1)*(2*y - 1) - 2*y**2*(2*x - 1)*(y - 1) - 2*y*(2*x - 1)*(y - 1)**2) - 4.0*x**2*y**2*z**3*(x - 1)**2*(y - 1)**2*(z - 1)**3*(y*(y - 1)*(2*z - 1) - z*(2*y - 1)*(z - 1))*(-2*x**2*(x - 1)*(2*y - 1) + 2*x*y*(x - 1)*(y - 1) + x*y*(2*x - 1)*(y - 1) - 2*x*(x - 1)**2*(2*y - 1) + y*(x - 1)*(2*x - 1)*(y - 1)) + 4.0*x*y*z**2*(x - 1)*(y - 1)*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1)) + 16.0*x*y*z*(x - 1)*(y - 1)*(z - 1)*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1)) + 4.0*x*y*(x - 1)*(y - 1)*(z - 1)**2*(x*(x - 1)*(2*y - 1) - y*(2*x - 1)*(y - 1)) + x*z**2*(x - 1)*(z - 1)**2*(8.0*x*y*(x - 1) + 8.0*x*(x - 1)*(y - 1) + 4.0*x*(x - 1)*(2*y - 1) + y**2*(4.0 - 8.0*x) - 16.0*y*(2*x - 1)*(y - 1) + (4.0 - 8.0*x)*(y - 1)**2) + y*z**2*(y - 1)*(z - 1)**2*(x**2*(8.0*y - 4.0) - 8.0*x*y*(y - 1) + 16.0*x*(x - 1)*(2*y - 1) + 4.0*y*(1 - 2*x)*(y - 1) + 8.0*y*(1 - x)*(y - 1) + (x - 1)**2*(8.0*y - 4.0)) + 2*(2*x - 1)*(2*y - 1)
        return result
    
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = x**2*(x-1)**2*(2*y*(y-1)*(2*y-1)*z**2*(z-1)**2 - y**2*(y-1)**2*2*z*(z-1)*(2*z-1))
        result[..., 1] = y**2*(y-1)**2*(2*z*(z-1)*(2*z-1)*x**2*(x-1)**2 - z**2*(z-1)**2*2*x*(x-1)*(2*x-1))
        result[..., 2] = z**2*(z-1)**2*(2*x*(x-1)*(2*x-1)*y**2*(y-1)**2 - x**2*(x-1)**2*2*y*(y-1)*(2*y-1))
        return result
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return (2*x-1)*(2*y-1)*(2*z-1)
    
    @cartesian
    def velocity_dirichlet(self, p):
        return self.velocity(p)
    
    @cartesian
    def pressure_dirichlet(self, p):
        return self.pressure(p)
