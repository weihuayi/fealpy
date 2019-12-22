import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData

from fealpy.wg.SobolevEquationWGModel2d import SobolevEquationWGModel2d

from fealpy.boundarycondition import DirichletBC

from fealpy.mesh.simple_mesh_generator import triangle

from fealpy.pde.sobolev_equation_2d import SinSinExpData
from fealpy.solver import MatlabSolver

class SobolevEquationWGModel2dTest:
    def __init__(self):
        nu = 1
        epsilon = 0.1
        self.pde = SinSinExpData(nu, epsilon)
        self.solver = MatlabSolver()

    def test_poisson_equaion(self, p=1, maxit=4):
        h = 0.1
        pde = CosCosData()
        domain = pde.domain()
        error = np.zeros((maxit,), dtype=np.float)
        for i in range(maxit):
            mesh = triangle(domain, h, meshtype='polygon')
            dmodel = SobolevEquationWGModel2d(self.pde, mesh, p=p)
            S = dmodel.space.stabilizer_matrix()
            A = dmodel.G + S
            F = dmodel.space.source_vector(pde.source)
            A, F = dmodel.space.apply_dirichlet_bc(pde.dirichlet, A, F)
            uh = dmodel.space.function()
            uh[:] = spsolve(A, F)
            integralalg = dmodel.space.integralalg
            error[i] = integralalg.L2_error(pde.solution, uh)
            h /= 2

        print(error)
        print(error[0:-1]/error[1:])

    def test_sobolev_equation(self, p=1, maxit=4):
        pde = self.pde
        domain = pde.domain()
        timeline = pde.time_mesh(0, 1, 20)
        error = np.zeros(maxit, dtype=np.float)
        h = 0.1
        for i in range(maxit):
            print(i)
            mesh = triangle(domain, h=h, meshtype='polygon')
            dmodel = SobolevEquationWGModel2d(pde, mesh, p)
            uh = dmodel.init_solution(timeline)
            up = dmodel.projection(pde.solution, timeline)
            fp = dmodel.projection(pde.source, timeline)
            uh[:, 0] = up[:, 0]
            solver = self.solver.divide
            data = [uh, fp, solver]
            timeline.time_integration(data, dmodel, solver)
            error[i] = np.max(np.abs(uh - up))

            timeline.uniform_refine()
            h /= 2

        print(error[:-1]/error[1:])
        print(error)


test = SobolevEquationWGModel2dTest()
#test.test_poisson_equaion()
test.test_sobolev_equation()
