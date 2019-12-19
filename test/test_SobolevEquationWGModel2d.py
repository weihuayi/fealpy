import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.sobolev_equation_2d import SinSinExpData

from fealpy.wg.SobolevEquationWGModel2d import SobolevEquationWGModel2d

from fealpy.boundarycondition import DirichletBC

from fealpy.mesh.simple_mesh_generator import triangle


class SobolevEquationWGModel2dTest:
    def __init__(self):
        nu = 1
        epsilon = 0.1
        self.pde = SinSinExpData(nu, epsilon)

    def test_poisson_equaion(self, maxit=4):
        h = 0.1
        pde = CosCosData()
        domain = pde.domain()
        error = np.zeros((maxit,), dtype=np.float)
        for i in range(maxit):
            mesh = triangle(domain, h, meshtype='polygon')
            dmodel = SobolevEquationWGModel2d(self.pde, mesh, p=2)
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

    def test_sobolev_equation(self, maxit=4):
        pass


test = SobolevEquationWGModel2dTest()
test.test_poisson_equaion()
