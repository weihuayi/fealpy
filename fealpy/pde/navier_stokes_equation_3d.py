#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_equation_3d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Wed 04 Dec 2024 03:56:15 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import TetrahedronMesh

class ChannelFlow: 
    def __init__(self, eps=1e-10, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu
    
    def mesh(self, n=16):
        box = [0, 1, 0, 1, 0, 1]
        mesh = TetrahedronMesh.from_cylinder_gmsh(0.5, 3, 0.1)
        return mesh
    
    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        value = np.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def is_p_boundary(self, p):
        tag_left = np.abs(p[..., 0]) < self.eps
        tag_right = np.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_wall_boundary(self, p):
        tag_up = np.abs(p[..., 2] - 1.0) < self.eps
        tag_down = np.abs(p[..., 2]) < self.eps
        tag_front = np.abs(p[..., 1] - 1) < self.eps
        tag_back = np.abs(p[..., 1]) < self.eps
        return tag_up | tag_down | tag_front | tag_back


