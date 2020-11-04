#!/usr/bin/env python3
# 
import sys

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spilu
from scipy.sparse import spdiags

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import  BoxDomainData3d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import CrouzeixRaviartFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

import pyamg
from timeit import default_timer as timer


n = int(sys.argv[1])

pde = BoxDomainData3d() 
mesh = pde.init_mesh(n=n)


space = LagrangeFiniteElementSpace(mesh, p=1)
bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
uh = space.function(dim=3)
A = space.linear_elasticity_matrix(pde.lam, pde.mu, q=1)
F = space.source_vector(pde.source, dim=3)
A, F = bc.apply(A, F, uh)

if False:
    uh.T.flat[:] = spsolve(A, F)
elif False:
    N = len(F)
    print(N)
    start = timer()
    ilu = spilu(A.tocsc(), drop_tol=1e-6, fill_factor=40)
    end = timer()
    print('time:', end - start)

    M = LinearOperator((N, N), lambda x: ilu.solve(x))
    start = timer()
    uh.T.flat[:], info = cg(A, F, tol=1e-8, M=M)   # solve with CG
    print(info)
    end = timer()
    print('time:', end - start)
elif True:
    I = space.rigid_motion_matrix()
    P = space.stiff_matrix(c=2*pde.mu)
    isBdDof = space.set_dirichlet_bc(uh, pde.dirichlet,
            threshold=pde.is_dirichlet_boundary)
    solver = LinearElasticityLFEMFastSolver(A, P, I, isBdDof) 
    start = timer()
    uh[:] = solver.solve(uh, F) 
    end = timer()
    print('time:', end - start, 'dof:', A.shape)
elif False:
    A0 = space.stiff_matrix(c=2*pde.mu)
    isBdDof = space.set_dirichlet_bc(uh, pde.dirichlet,
            threshold=pde.is_dirichlet_boundary)
    P = space.rigid_motion_matrix()
    solver = LinearElasticityLFEMFastSolver_2(A, A0, P, isBdDof)
    start = timer()
    uh[:] = solver.solve(uh, F) 
    end = timer()
    print('time:', end - start, 'dof:', A.shape)
else:
    aspace = CrouzeixRaviartFiniteElementSpace(mesh)
    I = aspace.interpolation_matrix()
    P = aspace.linear_elasticity_matrix(pde.lam, pde.mu)
    isBdDof = aspace.is_boundary_dof(threshold=pde.is_dirichlet_boundary)
    isBdDof = np.r_['0', isBdDof, isBdDof, isBdDof]

    solver = LinearElasticityLFEMFastSolver_1(A, I, P, isBdDof) 
    start = timer()
    uh[:] = solver.solve(uh, F) 
    end = timer()
    print('time:', end - start, 'dof:', A.shape)


if False:
# 原始网格
    mesh.add_plot(plt)

# 变形网格
    mesh.node += scale*uh
    mesh.add_plot(plt)

    plt.show()
