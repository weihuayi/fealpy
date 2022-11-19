import numpy as np

from fealpy.functionspace import LagrangeFiniteElementSpace
from ..boundarycondition import DirichletBC
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class PoissonFEMModel(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = LagrangeFiniteElementSpace(mesh, p, q=q)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.integrator = self.mesh.integrator(p+2)
    def recover_estimate(self, rguh):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.space.cellmeasure
        return np.sqrt(e)

    def residual_estimate(self, uh=None):
        if uh is None:
            uh = self.uh
        mesh = self.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        n = mesh.face_normal()
        bc = np.array([1/(GD+1)]*(GD+1), dtype=mesh.ftype)
        grad = uh.grad_value(bc)

        ps = mesh.bc_to_point(bc)
        try:
            d = self.pde.diffusion_coefficient(ps)
        except AttributeError:
            d = np.ones(NC, dtype=mesh.ftype)

        if isinstance(d, float):
            grad *= d
        elif len(d) == GD:
            grad = np.einsum('m, im->im', d, grad)
        elif isinstance(d, np.ndarray):
            if len(d.shape) == 1:
                grad = np.einsum('i, im->im', d, grad)
            elif len(d.shape) == 2:
                grad = np.einsum('im, im->im', d, grad)
            elif len(d.shape) == 3:
                grad = np.einsum('imn, in->in', d, grad)

        if GD == 2:
            face2cell = mesh.ds.edge_to_cell()
            h = np.sqrt(np.sum(n**2, axis=-1))
        elif GD == 3:
            face2cell = mesh.ds.face_to_cell()
            h = np.sum(n**2, axis=-1)**(1/4)

        J = h*np.sum((grad[face2cell[:, 0]] - grad[face2cell[:, 1]])*n, axis=-1)**2

        NC = mesh.number_of_cells()
        eta = np.zeros(NC, dtype=mesh.ftype)
        np.add.at(eta, face2cell[:, 0], J)
        np.add.at(eta, face2cell[:, 1], J)
        return np.sqrt(eta)

    def get_left_matrix(self):
        return self.space.stiff_matrix()

    def get_right_vector(self):
        return self.space.source_vector(self.pde.source)

    def solve(self):
        bc = DirichletBC(self.space, self.pde.dirichlet)

        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()
        self.A = A

        print("Construct linear system time:", end - start)

        AD, b = bc.apply(A, b)

        start = timer()
        self.uh[:] = spsolve(AD, b)
        end = timer()
        print("Solve time:", end-start)

        ls = {'A': AD, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self, uh=None):
        u = self.pde.solution
        if uh is None:
            uh = self.uh.value
        else:
            uh = uh.value
        L2 = self.space.integralalg.L2_error(u, uh)
        return L2

    def H1_semi_error(self, uh=None):
        gu = self.pde.gradient
        if uh is None:
            guh = self.uh.grad_value
        else:
            guh = uh.grad_value
        H1 = self.space.integralalg.L2_error(gu, guh)
        return H1

    def recover_error(self, rguh):
        gu = self.pde.gradient
        guh = rguh.value
        mesh = self.mesh
        re = self.space.integralalg.L2_error(gu, guh, mesh)
        return re
