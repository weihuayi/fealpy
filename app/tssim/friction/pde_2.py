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

class ChannelFlow: 
    def __init__(self, eps=1e-10, rho=1, mu=1, R=None):
        self.eps = eps
        self.rho = rho
        self.mu = mu
        if R is None:
            self.R = rho/mu
    
    def mesh(self, n=16):
        box = [0, 10, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        return mesh
    
    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 10.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down 



















class CouetteFlow2:
    '''
    @brief: CouetteFlow
    '''

    def __init__(self, eps=1e-10, h=1/256):
        self.eps = eps
        
        ## init the parameter
        self.R = 3.0 ##dimensionless
        self.l_s = 0.0038 ##dimensionless slip length
        #self.L_s = self.l_s / h 
        #self.L_s = 1/self.l_s 
        self.L_s = self.l_s
        self.epsilon = 0.001 ## the thickness of interface
        self.L_d = 5e-4 ##phenomenological mobility cofficient
        self.lam = 12.0 ##dimensionless
        self.V_s = 500.0 ##dimensionless 
        self.s = 1.5 ##stablilizing parameter
        self.theta_s = bm.array(bm.pi/2)
        self.h = h

    def mesh(self):
        box = [0, 1, 0, 0.4]
        mesh = TriangleMesh.from_box(box, nx=1024, ny=512)
        return mesh

    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] - 0.4) < self.eps) | \
               (bm.abs(p[..., 1] - 0.0) < self.eps)
    
    @cartesian
    def is_up_boundary(self,p):
        return bm.abs(p[..., 1] - 0.4) < self.eps
    
    @cartesian
    def is_down_boundary(self,p):
        return bm.abs(p[..., 1] - 0.0) < self.eps
    
    @cartesian
    def is_uy_Dirichlet(self,p):
        return (bm.abs(p[..., 1] - 0.4) < self.eps) | \
               (bm.abs(p[..., 1] - 0.0) < self.eps)
    
    @cartesian
    def init_phi(self,p):
        x = p[..., 0]
        y = p[..., 1]   
        tagfluid0 = x > 0.2
        tagfluid1 = x <= 0.2 
        phi = bm.zeros_like(x)
        phi[tagfluid0] = 1.0
        phi[tagfluid1] = -1.0
        return phi

    @cartesian        
    def u_w(self, p):
        y = p[..., 1]
        result = bm.zeros_like(p)
        tag_up = (bm.abs(y-0.4)) < self.eps 
        tag_down = (bm.abs(y-0.0)) < self.eps 
        result[..., 0] = bm.where(tag_up, 0.2, 0) + bm.where(tag_up, 0.2, 0)
        return result

    @cartesian
    def p_dirichlet(self, p):
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_p_dirichlet(self, p):
        return bm.zeros_like(p[..., 0], dtype=bool)
