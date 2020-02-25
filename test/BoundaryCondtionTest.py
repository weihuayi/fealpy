#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

class BoundaryConditionTest:
    def __init__(self):
        pass

    def poisson_fem_2d_test(self, p=1):
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

    def poisson_fem_2d_neuman_test(self, p=1):
        pde = CosCosData()
        mesh = pde.init_mesh(n=2)
        for i in range(4):
            space = LagrangeFiniteElementSpace(mesh, p=p)
            A = space.stiff_matrix()
            b = space.source_vector(pde.source)
            uh = space.function()
            bc = BoundaryCondition(space, neuman=pde.neuman)
            bc.apply_neuman_bc(b)
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
#test.poisson_fem_2d_test(p=2)
test.poisson_fem_2d_neuman_test(p=3)
