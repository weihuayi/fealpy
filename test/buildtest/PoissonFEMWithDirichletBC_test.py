#!/usr/bin/env python3
# 

import sys 

import numpy as np


from fealpy.pde.poisson_2d import CosCosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 


from scipy.sparse.linalg import spsolve


dim = int(sys.argv[1])

def test_poisson_fem(dim):
    degree = 1  
    nrefine = 4 
    maxit = 4 

    if dim == 2:
        from fealpy.pde.poisson_2d import CosCosData as PDE
    elif dim == 3:
        from fealpy.pde.poisson_3d import CosCosCosData as PDE

    pde = PDE()
    mesh = pde.init_mesh(n=nrefine)

    errorMatrix = np.zeros((2, maxit), dtype=np.float64)
    NDof = np.zeros(maxit, dtype=np.int64)

    for i in range(maxit):
        space = LagrangeFiniteElementSpace(mesh, p=degree)
        NDof[i] = space.number_of_global_dofs()
        bc = DirichletBC(space, pde.dirichlet) 

        uh = space.function()
        if dim == 2:
            A = space.stiff_matrix()
        elif dim == 3:
            A = space.parallel_stiff_matrix(q=p)

        F = space.source_vector(pde.source)

        A, F = bc.apply(A, F, uh)

        uh[:] = spsolve(A, F).reshape(-1)

        errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
        errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

        if i < maxit-1:
            mesh.uniform_refine()

