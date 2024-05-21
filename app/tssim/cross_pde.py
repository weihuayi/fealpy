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
        self.rho = 0.001
        self.c = 0.588
        self.lam = 0.139

    def domain(self):
        return self.box
    
    @cartesian
    def is_outflow_boundary(p):
        return np.abs(p[..., 0]-self.box[1])<eps

    @cartesian
    def is_left_boundary(p):
        return np.abs(p[..., 0]-self.box[0])<eps
    
    @cartesian
    def is_inlet_boundary(p):
        return (np.abs(p[..., 0]-self.box[0])<eps)|\
               (np.abs(p[..., 1]-0.5)<0.25)

    @cartesian
    def is_wall_boundary(p):
        return (np.abs(p[..., 1]-self.box[3])<eps)|\
               (np.abs(p[..., 1]-self.box[4])<eps)
   

