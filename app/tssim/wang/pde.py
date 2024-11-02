#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: pde.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 03:49:34 PM CST
	@bref 
	@ref 
'''  
from fealpy.decorator import barycentric,cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

class CouetteFlow:
    '''
    @brief: CouetteFlow
    '''

    def __init__(self, eps=1e-10):
        self.eps = eps
        
        ## init the parameter
        self.R = 5 ##dimensionless
        self.L_s = 0.0025 ##dimensionless slip length
        self.epsilon = 0.004 ## the thickness of interface
        self.lam = 12.0 ##dimensionless
        self.V_s = 200 ##dimensionless 
        self.L_d = 0.0005 ##phenomenological mobility cofficient
        self.s = 2.5 ##stablilizing parameter
        self.theta_s = 90

    def mesh(self, h=1/256):
        box = [-0.5, 0.5, -0.125, 0.125]
        mesh = TriangleMesh.from_box(box, nx=int(1/h), ny=int(0.25/h))
        return mesh

    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] - 0.125) < self.eps) | \
               (bm.abs(p[..., 1] + 0.125) < self.eps)
    
    @cartesian
    def init_phi(self,p):
        x = p[..., 0]
        y = p[..., 1]   
        tagfluid = bm.logical_and(x > -0.25, x < 0.25)
        tagwall = bm.logical_not(tagfluid)
        phi = bm.zeros_like(x)
        phi[tagfluid] = 1.0
        phi[tagwall] = -1.0
        return phi

    @cartesian        
    def u_w(self, p):
        x = p[..., 0]
        y = p[..., 1]
        u = bm.zeros_like(p)
        tag_up = bm.abs(y-0.125) < self.eps
        tag_down = bm.abs(y+0.125) < self.eps
        u[tag_up, 0] = 0.2
        u[tag_down, 0] = -0.2
        return u
