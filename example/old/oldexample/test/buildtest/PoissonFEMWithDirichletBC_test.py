#!/usr/bin/env python3
# 

import sys 

import numpy as np


from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 


from scipy.sparse.linalg import spsolve


def test_poisson_fem_2d():
    degree = 1  
    dim = 2
    nrefine = 4 
    maxit = 4 

    from fealpy.pde.poisson_2d import CosCosData as PDE

    pde = PDE()
    mesh = pde.init_mesh(n=nrefine)

    errorMatrix = np.zeros((2, maxit), dtype=np.float64)
    NDof = np.zeros(maxit, dtype=np.int64)

    for i in range(maxit):
        space = LagrangeFiniteElementSpace(mesh, p=degree)
        NDof[i] = space.number_of_global_dofs()
        bc = DirichletBC(space, pde.dirichlet) 

        uh = space.function()
        A = space.stiff_matrix()

        F = space.source_vector(pde.source)

        A, F = bc.apply(A, F, uh)

        uh[:] = spsolve(A, F).reshape(-1)

        errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
        errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

        if i < maxit-1:
            mesh.uniform_refine()

