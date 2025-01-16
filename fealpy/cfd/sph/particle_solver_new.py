#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: particle_solver_new.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 14 Jan 2025 03:44:28 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

# Types
Box = TensorLike
f32 = bm.float32

class SPHSolver:
    def __init__(self, mesh):
        self.mesh = mesh 


    def pairwise_displacement(self, Ra: TensorLike, Rb: TensorLike):
        if len(Ra.shape) != 1:
            msg = (
				"Can only compute displacements between vectors. To compute "
				"displacements between sets of vectors use vmap or TODO."
				)
            raise ValueError(msg)

        if Ra.shape != Rb.shape:
            msg = "Can only compute displacement between vectors of equal dimension."
            raise ValueError(msg)

        return Ra - Rb

    def periodic_displacement(self, side: Box, dR: TensorLike):

        return bm.mod(dR + side * f32(0.5), side) - f32(0.5) * side


'''
	def periodic(side: Box):
		def displacement_fn
		pass
'''