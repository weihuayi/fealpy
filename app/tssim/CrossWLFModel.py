#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross_pde.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 18 May 2024 03:01:06 PM CST
	@bref 
	@ref 
'''  

import numpy as np
from fealpy.decorator import barycentric, cartesian
class CrossWLF:
    def __init__(self):
        self.box = [0, 10, 0, 1]
        self.eps = 1e-10
        self.rho_l = 1020
        self.lambda_l = 0.173
        self.U = 1
        self.L = 0.25
        self.T0 = 1
        self.rho_l = 1020
        self.rho = 0.001
        self.c = 0.588
        self.lam = 0.139
        self.eta_g = 1.792e-5
        self.WE = 8995
        self.Pe = 2505780.347

    def domain(self):
        return self.box
    
    @cartesian
    def is_outflow_boundary(self, p):
        return np.abs(p[..., 0]-self.box[1])<self.eps

    @cartesian
    def is_left_boundary(self,p):
        return np.abs(p[..., 0]-self.box[0])<self.eps
    
    @cartesian
    def is_inlet_boundary(self, p):
        return (np.abs(p[..., 0]-self.box[0])<self.eps) &\
               (np.abs(p[..., 1]-0.5)<0.25)

    @cartesian
    def is_wall_boundary(self ,p):
        return (np.abs(p[..., 1]- self.box[2])<self.eps)|\
               (np.abs(p[..., 1]- self.box[3])<self.eps)
   
    @cartesian
    def init_surface(self, p):
        x = p[...,0]
        y = p[...,1]
        val =  -x
        return val
    
    @cartesian
    def boundary_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x
        val[:] = 0
        return val
    
