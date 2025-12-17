from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import CylinderMesher

from ....simulation.time import UniformTimeLine
import sympy as sp

class Exp0004(CylinderMesher):
    def __init__(self, options: dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.radius = 0.5
        self.height = 3.0
        self.lc = 0.4
        super().__init__(radius=self.radius, height=self.height, lc=self.lc)
        self.mesh = self.init_mesh()

    
    @cartesian
    def velocity(self, p, t):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        value = bm.zeros(p.shape)
        value[...,2] = 4 * (0.25 - y**2 - x**2)
        return value
    
    @cartesian
    def pressure(self, p, t):
        z = p[..., 2]
        val = 8*(1-z) 
        # val = bm.zeros(z.shape)
        return val
    
    @cartesian
    def source(self, p, t):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        f = bm.zeros(p.shape)
        f[..., 2] = 16
        return f
    
    @cartesian
    def is_pressure_boundary(self, p = None):
        # if p is None:
        #     return 1
        # tag_left = bm.abs(p[..., 2]) < self.eps
        # tag_right = bm.abs(p[..., 2] - 1.0) < self.eps
        # return tag_left | tag_right
        return 0

    @cartesian
    def is_velocity_boundary(self, p):
        # x = p[..., 0]
        # y = p[..., 1]
        # z = p[..., 2]
        # r = self.radius
        # tag = bm.abs(x**2 + y**2 - r**2) < 1e-3
        # return tag
        return None
    
    @cartesian
    def velocity_dirichlet(self, p, t):
        return self.velocity(p, t)
    
    @cartesian
    def pressure_dirichlet(self, p, t):
        return self.pressure(p, t)

