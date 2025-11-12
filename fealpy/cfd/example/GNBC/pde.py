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

    def __init__(self, eps=1e-10, h=1/256):
        self.eps = eps
        
        ## init the parameter
        self.R = 5.0 ##dimensionless
        self.l_s = 0.0025 ##dimensionless slip length
        self.L_s = self.l_s / 1000

        self.epsilon = 0.004 ## the thickness of interface
        self.L_d = 0.0005 ##phenomenological mobility cofficient
        self.lam = 12.0 ##dimensionless
        self.V_s = 200.0 ##dimensionless 
        self.s = 2.5 ##stablilizing parameter
        #self.theta_s = bm.array(bm.pi/2)
        self.theta_s = bm.array(77.6/180 * bm.pi)
        self.h = h

    def mesh(self):
        box = [-0.5, 0.5, -0.125, 0.125]
        mesh = TriangleMesh.from_box(box, nx=int(1/self.h), ny=int(0.25/self.h))
        return mesh

    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] - 0.125) < self.eps) | \
               (bm.abs(p[..., 1] + 0.125) < self.eps)
    
    @cartesian
    def is_up_boundary(self,p):
        return bm.abs(p[..., 1] - 0.125) < self.eps
    
    @cartesian
    def is_down_boundary(self,p):
        return bm.abs(p[..., 1] + 0.125) < self.eps
    
    @cartesian
    def is_uy_Dirichlet(self,p):
        return (bm.abs(p[..., 1] - 0.125) < self.eps) | \
               (bm.abs(p[..., 1] + 0.125) < self.eps)
    
    @cartesian
    def init_phi(self,p):
        x = p[..., 0]
        y = p[..., 1]   
        tagfluid0 = bm.logical_and(x > -0.25, x < 0.25)
        tagfluid1 = bm.logical_not(tagfluid0)
        phi = bm.zeros_like(x)
        phi[tagfluid0] = 1.0
        phi[tagfluid1] = -1.0
        return phi

    @cartesian        
    def u_w(self, p):
        y = p[..., 1]
        result = bm.zeros_like(p)
        tag_up = (bm.abs(y-0.125)) < self.eps 
        tag_down = (bm.abs(y+0.125)) < self.eps
        value = bm.where(tag_down, -0.2, 0) + bm.where(tag_up, 0.2, 0)
        result[..., 0] = value 
        return result

    @cartesian
    def p_dirichlet(self, p):
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_p_dirichlet(self, p):
        return bm.zeros_like(p[..., 0], dtype=bool)




















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
