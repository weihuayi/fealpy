#!/usr/bin/env python3
# 
# sudo -H pip3 install transplant

import sys
from timeit import default_timer as timer

import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.solver import MatlabSolver
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

p = int(sys.argv[1])
maxit = int(sys.argv[2])

start = timer()
solver = MatlabSolver()
end = timer()

print("The matalb start time:", end - start)


pde = CosCosData()
mesh = pde.init_mesh(4)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=1)
    gdof = space.number_of_global_dofs()
    print("The num of dofs:", gdof)
    A = space.stiff_matrix()
    b = space.source_vector(pde.source)
    bc = DirichletBC(space, pde.dirichlet)
    AD, b = bc.apply(A, b)
    uh0 = solver.divide(AD, b)
    start = timer()
    uh1 = spsolve(AD, b)
    end = timer()
    print("The spsolver time:", end - start)
    
    print(np.sum(np.abs(uh0 - uh1)))
    #uh1 = solver.mumps_solver(A, b)
    #uh2 = solver.ifem_amg_solver(A, b)
    if i < maxit-1:
        mesh.uniform_refine()

