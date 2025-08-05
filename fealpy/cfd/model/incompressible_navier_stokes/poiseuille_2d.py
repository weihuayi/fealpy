import sympy as sp
from typing import Union, Callable, Dict

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from fealpy.mesh import TriangleMesh
from fealpy.mesher.box_mesher import BoxMesher2d

CoefType = Union[int, float, Callable]

class Poiseuille2D(BoxMesher2d):
    def __init__(self, eps=1e-10, rho=1, mu=1, R=None):
        self.box = [0,1,0,1]
        self.eps = eps
        self.rho = rho
        self.mu = mu
        if R is None:
            self.R = rho/mu
    
    @cartesian
    def velocity(self, p, t):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        if t != 0:
            value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        if t == 0:
            val = bm.zeros_like(x, dtype=p.dtype)
        else:
            val = 8*(1-x) 
        return val
    
    @cartesian
    def is_pressure_boundary(self, p):
        tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
        tag_right = bm.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right
    
    @cartesian
    def is_velocity_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down
    
    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape)
        return val   
    
    velocity_dirichlet = velocity
    pressure_dirichlet = pressure 

