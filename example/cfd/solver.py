#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 14 Oct 2024 06:49:22 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.pde.poisson_2d import CosCosData
from  mumps import DMumpsContext

import scipy.sparse as sp
import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve
def scipy_solve(A, b):
    A =  A.to_scipy() 
    b = bm.to_numpy(b)
    x = sp.linalg.spsolve(A,b)
    return x

def mumps_solve(A, b):
    A = A.to_scipy()
    b = bm.to_numpy(b)
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)
    ctx.set_rhs(b)
    ctx.run(job=6)
    ctx.destroy()
    return b

def cupy_solve(A, b):
    A =  A.to_scipy() 
    A = cp.sparse.csr_matrix(A.astype(cp.float64)) 
    b = bm.to_numpy(b)
    b = cp.array(b, dtype=b.dtype)
    x = spsolve(A,b)
    return b


bm.set_backend('pytorch')
pde = CosCosData()
mesh = TriangleMesh.from_box([0,1,0,1], nx=10, ny=10)
space= LagrangeFESpace(mesh, p=1)
uh = space.function()  # 建立一个有限元函数

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator())
lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source))

A = bform.assembly()
F = lform.assembly()

gdof = space.number_of_global_dofs()
A, F = DirichletBC(space, gd=pde.solution).apply(A, F)

x = scipy_solve(A, F)
#x = mumps_solve(A, F)
#x = cupy_solve(A, F)




