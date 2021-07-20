#!/usr/bin/env python3
# 
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
import transplant

from fealpy.pde.poisson_2d import CosCosData
from fealpy.wg.SobolevEquationWGModel2d import SobolevEquationWGModel2d
from fealpy.boundarycondition import DirichletBC
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.pde.sobolev_equation_2d import SinSinExpData, PolyExpData
from fealpy.tools import show_error_table

class SobolevEquationWGModel2dTest:
    def __init__(self):
        mu = 1
        epsilon = 0.1
        self.pde = SinSinExpData(mu, epsilon)
        matalb = transplant.Matlab()        
        self.solver = MatlabSolver(matlab)

    def test_poisson_equation(self, p=1, maxit=4):
        h = 0.1
        pde = CosCosData()
        domain = pde.domain()
        error = np.zeros((maxit,), dtype=np.float)
        for i in range(maxit):
            mesh = triangle(domain, h, meshtype='polygon')
            dmodel = SobolevEquationWGModel2d(self.pde, mesh, p=p)
            uh = dmodel.space.function()
            dmodel.space.set_dirichlet_bc(uh, pde.dirichlet)

            S = dmodel.space.stabilizer_matrix()
            A = dmodel.G + S
            F = dmodel.space.source_vector(pde.source)
            F -= A@uh

            isBdDof = dmodel.space.boundary_dof()
            gdof = dmodel.space.number_of_global_dofs()
            bdIdx = np.zeros(gdof, dtype=np.int)
            bdIdx[isBdDof] = 1
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            A = T@A@T + Tbd

            F[isBdDof] = uh[isBdDof]
            uh[:] = spsolve(A, F)
            integralalg = dmodel.space.integralalg
            error[i] = integralalg.L2_error(pde.solution, uh)
            h /= 2

        print(error)
        print(error[0:-1]/error[1:])

    def test_sobolev_equation(self, p=1, maxit=5):
        mu = 1
        epsilon = 0.1
        pde = PolyExpData(mu, epsilon)
        domain = pde.domain()
        h = 0.25
        timeline = pde.time_mesh(0, 0.1, 160)
        errorType = ['$|| u - u_{h,i}||_{0}$',
             '$|| p - p_{h, i}||_{0}$',
             '$||\\nabla u - \\nabla_w u_h||_{0}$',
             '$||\\nabla\cdot p - \\nabla_w\cdot p_h||_{0}$',
             '$||\\nabla u - \\nabla u_{h, i}||_{0}',
             '$||\\nabla\cdot p - \\nabla\cdot p_{h, i}||_{0}']
        error = np.zeros((len(errorType), maxit), dtype=np.float)
        Ndof = np.zeros(maxit, dtype=np.int)
        for i in range(maxit):
            print(i)
            mesh = triangle(domain, h=h, meshtype='polygon')
            dt = timeline.current_time_step_length()
            dmodel = SobolevEquationWGModel2d(pde, mesh, p, q=6, dt=dt, step=50)

            uh = dmodel.space.project(lambda x:pde.solution(x, 0.0))
            solver = self.solver.divide

            data = [uh, 0, solver, error[:, i]]

            timeline.time_integration(data, dmodel, solver)

            Ndof[i] = dmodel.space.number_of_global_dofs()
            h /= 2
            timeline.uniform_refine(n=2)

        error = np.sqrt(error)
        show_error_table(Ndof, errorType, error)


test = SobolevEquationWGModel2dTest()
#test.test_poisson_equation(p=2)
test.test_sobolev_equation(maxit=5)
