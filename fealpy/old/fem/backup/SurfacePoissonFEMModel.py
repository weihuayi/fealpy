import numpy as np
import transplant

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from ..quadrature  import TriangleQuadrature
from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
from fealpy.solver import MatlabSolver
from ..boundarycondition import DirichletBC
from fealpy.quadrature.FEMeshIntegralAlg import FEMeshIntegralAlg

from scipy.sparse.linalg import spsolve
class SurfacePoissonFEMModel():
    def __init__(self, mesh, pde, p, q, p0=None):
        """
        """
        self.space = SurfaceLagrangeFiniteElementSpace(mesh, pde.surface, p=p, p0=p0)
        self.mesh = self.space.mesh
        self.surface = pde.surface
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        #matlab = transplant.Matlab()
        #self.solver = MatlabSolver(matlab)

    def recover_estimate(self, rguh):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.area
        return np.sqrt(e)
    
    def get_grad_basis(self, bc):
        return self.space.grad_basis(bc, returncond=True)

    def get_left_matrix(self):
        return self.space.stiff_matrix()

    def get_right_vector(self):
        b = self.space.source_vector(self.pde.source)
        #b -= np.mean(b)
        return b

    def solve(self):
        #u = self.pde.solution
        #bc = DirichletBC(self.space, u, self.is_boundary_dof)
        A = self.get_left_matrix()
        b = self.get_right_vector()
        #AD, b = bc.apply(A, b)
        #self.uh[:] = self.solver.divide(A, b)
        self.uh[:] = spsolve(A, b)

    def is_boundary_dof(self, p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool_)
        isBdDof[0] = True
        return isBdDof

    def error(self):
        e0 = self.L2_error()
        e1 = self.H1_semi_error()
        return e0, e1

    def L2_error(self):
        u = self.pde.solution
        uh = self.uh.value
        return self.space.integralalg.L2_error(u, uh)

    def H1_semi_error(self):
        gu = self.pde.gradient
        guh = self.uh.grad_value
        return self.space.integralalg.L2_error(gu, guh)

