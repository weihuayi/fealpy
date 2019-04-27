import numpy as np

from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from .integral_alg import IntegralAlg

from ..boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve

from timeit import default_timer as timer


class PoissonFEMModel(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = LagrangeFiniteElementSpace(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = mesh.integrator(q)
        self.integralalg = IntegralAlg(
                self.integrator, self.mesh, self.cellmeasure)

    def recover_estimate(self, rguh):
        if self.space.p > 1:
            raise ValueError('This method only work for p=1!')
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.cellmeasure
        return np.sqrt(e)

    def get_left_matrix(self):
        return self.space.stiff_matrix(self.integrator, self.cellmeasure)

    def get_right_vector(self):
        return self.space.source_vector(self.pde.source, self.integrator, self.cellmeasure)

    def solve(self):
        bc = DirichletBC(self.space, self.pde.dirichlet)

        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()

        print("Construct linear system time:", end - start)

        AD, b = bc.apply(A, b)

        start = timer()
        self.uh[:] = spsolve(AD, b)
        end = timer()
        print("Solve time:", end-start)

        ls = {'A': AD, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def get_l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def get_L2_error(self, uh=None):
        u = self.pde.solution
        if uh is None:
            uh = self.uh.value
        else:
            uh = uh.value
        L2 = self.integralalg.L2_error(u, uh)
        return L2

    def get_H1_error(self, uh=None):
        gu = self.pde.gradient
        if uh is None:
            guh = self.uh.grad_value
        else:
            guh = uh.grad_value
        H1 = self.integralalg.L2_error(gu, guh)
        return H1

    def get_recover_error(self, rguh):
        gu = self.pde.gradient
        guh = rguh.value
        mesh = self.mesh
        re = self.integralalg.L2_error(gu, guh, mesh)
        return re
