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
        self.R = 1
    
    def mesh(self, lc):
        mesh = TetrahedronMesh.from_cylinder_gmsh(0.5, 1, lc)
        return mesh
    
    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        value = bm.zeros(p.shape)
        value[...,0] = 4*(0.25-y**2-x**2)
        return value
    
    @cartesian
    def pressure(self, p):
        z = p[..., 2]
        val = 8*(1-z) 
        return val
    
    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 2]) < self.eps
        tag_right = bm.abs(p[..., 2] - 3.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        tag = self.is_p_boundary(p)
        return ~tag


