from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class FlowPastFoil2D():
    def __init__(self, options : dict = None ):
        self.eps = 1e-10
        self.rho = 1.0
        self.mu = 0.001
        self.options = options

        if options is not None:
            self.box = options.get('box', [-0.5, 2.7, -0.4, 0.4])
            self.h = options.get('lc', 0.04)
        self.mesh = self.init_mesh()

    def init_mesh(self):
        from fealpy.mesher import NACA0012Mesher
        halos = bm.array([
        [1.0000, 0.00120],
        [0.9500, 0.01027],
        [0.9000, 0.01867],
        [0.8000, 0.03320],
        [0.7000, 0.04480],
        [0.6000, 0.05320],
        [0.5000, 0.05827],
        [0.4000, 0.06000],
        [0.3000, 0.05827],
        [0.2000, 0.05293],
        [0.1500, 0.04867],
        [0.1000, 0.04240],
        [0.0750, 0.03813],
        [0.0500, 0.03267],
        [0.0250, 0.02453],
        [0.0125, 0.01813],
        [0.0000, 0.00000],
        [0.0125, -0.01813],
        [0.0250, -0.02453],
        [0.0500, -0.03267],
        [0.0750, -0.03813],
        [0.1000, -0.04240],
        [0.1500, -0.04867],
        [0.2000, -0.05293],
        [0.3000, -0.05827],
        [0.4000, -0.06000],
        [0.5000, -0.05827],
        [0.6000, -0.05320],
        [0.7000, -0.04480],
        [0.8000, -0.03320],
        [0.9000, -0.01867],
        [0.9500, -0.01027],
        [1.0000, -0.00120],
        [1.00662, 0.0]], dtype=bm.float64)
        singular_points = bm.array([[0, 0], [1.00662, 0.0]], dtype=bm.float64)
        box = self.box
        h = self.h
        hs = [h/3, h/3]
        mesher = NACA0012Mesher(halos, box, singular_points)
        mesh = mesher.init_mesh(h, hs, is_quad=0, thickness=h/10, ratio=2.4, size=h/50)
        return mesh

    @cartesian
    def is_outflow_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        cond1 = bm.abs(x - self.box[1]) < self.eps
        cond2 = bm.abs(y-self.box[2])>self.eps
        cond3 = bm.abs(y-self.box[3])>self.eps
        return (cond1) & (cond2 & cond3) 
    
    @cartesian
    def is_inflow_boundary(self,p):
        return bm.abs(p[..., 0]-self.box[0]) < self.eps
     
    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] -self.box[2]) < self.eps) | \
               (bm.abs(p[..., 1] -self.box[3]) < self.eps)
    
    @cartesian
    def is_velocity_boundary(self,p):
        return ~self.is_outflow_boundary(p)
        # return None
    
    @cartesian
    def is_pressure_boundary(self,p=None):
        if p is None:
            return 1
        else:
            return self.is_outflow_boundary(p) 
            #return bm.zeros_like(p[...,0], dtype=bm.bool)
        # return 0

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(p)
        value[...,0] = 1.5*4*(y-self.box[2])*(self.box[3]-y)/(0.4**2)
        value[...,1] = 0
        return value
    
    @cartesian
    def pressure_dirichlet(self, p, t):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(x)
        return value

    @cartesian
    def velocity_dirichlet(self, p, t):
        x = p[...,0]
        y = p[...,1]
        index = self.is_inflow_boundary(p)
        result = bm.zeros_like(p)
        result[index] = self.u_inflow_dirichlet(p[index])
        return result
    
    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0
        result[..., 1] = 0
        return result
    
    def velocity(self, p ,t):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        return value

    def pressure(self, p, t):
        x = p[..., 0]
        val = bm.zeros_like(x)
        return val