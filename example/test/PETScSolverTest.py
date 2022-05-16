#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.solver.petsc_solver import PETScSolver


class PETScSolverTest():

    def __init__(self):
        pass


    def solve_poisson_3d(self, n=2):
        from fealpy.pde.poisson_3d import CosCosCosData as PDE
        from fealpy.mesh import MeshFactory
        from fealpy.functionspace import LagrangeFiniteElementSpace
        from fealpy.boundarycondition import DirichletBC 

        pde = PDE()
        mf = MeshFactory
        m = 2**n
        box = [0, 1, 0, 1, 0, 1]
        mesh = mf.boxmesh3d(box, nx=m, ny=m, nz=m, meshtype='tet')
        space = LagrangeFiniteElementSpace(mesh, p=1)
        gdof = space.number_of_global_dofs()
        NC = mesh.number_of_cells()
        print('gdof:', gdof, 'NC:', NC)
        bc = DirichletBC(space, pde.dirichlet) 
        uh = space.function()
        A = space.stiff_matrix()
        #A = space.parallel_stiff_matrix(q=1)
        
        #M = space.parallel_mass_matrix(q=2)
        M = space.mass_matrix()

        F = space.source_vector(pde.source)

        A, F = bc.apply(A, F, uh)

        solver = PETScSolver()
        print(111)
        solver.solve(A, F, uh)
        print(211)
        error = space.integralalg.L2_error(pde.solution, uh)
        print(error)



test = PETScSolverTest()

if sys.argv[1] == 'poisson3d':
    n = int(sys.argv[2])
    test.solve_poisson_3d(n=n)
