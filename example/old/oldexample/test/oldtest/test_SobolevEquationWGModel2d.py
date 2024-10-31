import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

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
test.test_poisson_equation()
#test.test_sobolev_equation()
