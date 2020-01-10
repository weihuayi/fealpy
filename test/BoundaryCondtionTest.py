#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

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
        mesh = pde.init_mesh(n=3)
        def is_dirichlet_boundary(p):
            flag = np.zeros(p.shape[0:-1], dtype=np.bool)
            flag[0] = True
            return flag
        for i in range(4):
            space = LagrangeFiniteElementSpace(mesh, p=p)
            A = space.stiff_matrix()
            b = space.source_vector(pde.source)
            uh = space.function()
            bc = BoundaryCondition(space, neuman=pde.neuman,
                    dirichlet=pde.dirichlet)
            bc.apply_neuman_bc(b)
            bc.apply_dirichlet_bc(A, b, uh,
                    is_dirichlet_boundary=is_dirichlet_boundary)
            uh[:] = spsolve(A, b).reshape(-1)
            error = space.integralalg.L2_error(pde.solution, uh)
            print(error)
            mesh.uniform_refine()


test = BoundaryConditionTest()
#test.poisson_fem_2d_test(p=2)
test.poisson_fem_2d_neuman_test(p=2)
