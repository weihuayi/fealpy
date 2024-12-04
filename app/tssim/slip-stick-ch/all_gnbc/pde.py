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

class slip_stick:

    def __init__(self, eps=1e-10, h=1/256):
        self.eps = eps
        
        ## init the parameter
        self.R = 5.0 ##dimensionless
        self.l_s = 0.0025 ##dimensionless slip length
        self.L_s = self.l_s

        self.epsilon = 0.004 ## the thickness of interface
        self.L_d = 0.0005 ##phenomenological mobility cofficient
        self.lam = 12.0 ##dimensionless
        self.V_s = 200.0 ##dimensionless 
        self.s = 2.5 ##stablilizing parameter
        self.theta_s = bm.array(90/180 * bm.pi)
        self.h = h

    def mesh(self):
        self.box = [0, 1, 0, 0.15]
        mesh = TriangleMesh.from_box(self.box, nx=int(1/self.h), ny=int(0.15/self.h))
        return mesh

    @cartesian
    def is_right_boundary(self, p):
        return bm.abs(p[..., 0]-self.box[1])<self.eps

    @cartesian
    def is_left_boundary(self,p):
        return bm.abs(p[..., 0]-self.box[0])<self.eps
    
    @cartesian
    def is_wall_boundary(self ,p):
        tag_up= bm.abs(p[..., 1]-self.box[3])<self.eps
        tag_down= bm.abs(p[..., 1]-self.box[2])<self.eps
        return tag_up | tag_down

    @cartesian
    def is_inout_boundary(self ,p):
        return (bm.abs(p[..., 0]- self.box[0])<self.eps)|\
               (bm.abs(p[..., 0]- self.box[1])<self.eps)
    
    @cartesian
    def init_interface(self,p):
        x = p[..., 0]
        y = p[..., 1]   
        tagfluid0 = x-0.05<self.eps
        tagfluid1 = bm.logical_not(tagfluid0)
        phi = bm.zeros_like(x)
        phi[tagfluid0] = 1.0
        phi[tagfluid1] = -1.0
        return phi

    
    @cartesian
    def boundary_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        tag0 = bm.abs(x-self.box[0])<self.eps
        tag1 = bm.abs(x-self.box[1])<self.eps
        value = bm.where(tag0, 2000, 0.0) + bm.where(tag1, 0.0, 0.0)
        value = bm.astype(value, bm.float64)
        return value
    
    @cartesian
    def is_stick_boundary(self, p):    
        return self.is_wall_boundary(p)
    '''
    @cartesian
    def is_stick_boundary(self, p):    
        x = p[..., 0]
        y = p[..., 1]
        domain = self.box
        #x_minmax_bounds = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],[0.9, 1.0]]
        #x_minmax_bounds2 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],[0.9, 1.0]]
        x_minmax_bounds = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        x_minmax_bounds2 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        #x_minmax_bounds = [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7],[0.8, 1]]
        #x_minmax_bounds2 = [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 1]]
        mn, mx = bm.array(x_minmax_bounds).T
        mn2 ,mx2 = bm.array(x_minmax_bounds2).T
        val = x[:, None]
        result1 = ((val > mn) & (val < mx) ).any(1)& (y-domain[2]<self.eps)
        result2 = ((val > mn2) & (val < mx2)).any(1) & (bm.abs(y-domain[3])<self.eps)
        result3 = ((y-domain[2]) < self.eps)|((y-domain[3]) < self.eps)
        return ((result1) | (result2)) & (result3)
    '''

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[..., 0]
        y = p[..., 1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        tag = self.is_left_boundary(p)
        value[tag, 0] = 100*2*y[tag]*(0.15-y[tag])
        value[tag, 1] = 0
        #value[...,0] = 100*2*y*(0.15-y)
        #value[..., 1] = 0
        value = bm.astype(value, bm.float64)
        return value
    
    @cartesian
    def is_uy_Dirichlet(self, p):
        tag0 = self.is_left_boundary(p)
        tag1 = self.is_wall_boundary(p)
        result = tag0 | tag1
        return result 
    
    @cartesian
    def is_ux_Dirichlet(self, p):
        tag0 = self.is_left_boundary(p)
        tag1 = self.is_wall_boundary(p)
        tag2 = self.is_stick_boundary(p)
        #result = tag0 | (tag1 & tag2)
        result = tag0
        return result 
