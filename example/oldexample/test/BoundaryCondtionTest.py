#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

import matplotlib.pyplot as plt


from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

class BoundaryConditionTest:
    def __init__(self):
        pass

    def poisson_fem_2d_dirichlet(self, p=1):
        pde = CosCosData()
        mesh = pde.init_mesh(n=3)
        for i in range(4):
            space = LagrangeFiniteElementSpace(mesh, p=p)
            A = space.stiff_matrix()
            b = space.source_vector(pde.source)
            uh = space.function()
            bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
            A, b = bc.apply_dirichlet_bc(A, b, uh)
            uh[:] = spsolve(A, b).reshape(-1)
            error = space.integralalg.L2_error(pde.solution, uh)
            print(error)
            mesh.uniform_refine()

    def poisson_fem_2d_neuman(self, p=1):
        pde = CosCosData()
        mesh = pde.init_mesh(n=2)
        for i in range(4):
            space = LagrangeFiniteElementSpace(mesh, p=p)
            A = space.stiff_matrix()
            b = space.source_vector(pde.source)
            uh = space.function()
            bc = BoundaryCondition(space, neumann=pde.neumann)
            bc.apply_neumann_bc(b)
            c = space.integral_basis()
            AD = bmat([[A, c.reshape(-1, 1)], [c, None]], format='csr')
            bb = np.r_[b, 0]
            x = spsolve(AD, bb)
            uh[:] = x[:-1]

            area = np.sum(space.integralalg.cellmeasure)
            ubar = space.integralalg.integral(pde.solution,
                    barycenter=False)/area
            def solution(p):
                return pde.solution(p) - ubar
            error = space.integralalg.L2_error(solution, uh)
            print(error)
            mesh.uniform_refine()


test = BoundaryConditionTest()
p = int(sys.argv[2])
if sys.argv[1] == 'dirichlet':
    test.poisson_fem_2d_dirichlet(p=p)

if sys.argv[1] == 'neumann':
    test.poisson_fem_2d_neuman(p=p)

