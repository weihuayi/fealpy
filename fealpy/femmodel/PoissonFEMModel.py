import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..solver.petsc_solver import linear_solver
from ..boundarycondition import DirichletBC
from ..femmodel import doperator 
from ..functionspace import FunctionNorm

class PoissonFEMModel(object):
    def __init__(self, mesh,  model, integrator=None, p=1):
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.mesh = self.V.mesh
        self.model = model
        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution)
        self.area = mesh.area()

        if integrator is None:
            self.integrator = TriangleQuadrature(3)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 

        self.error = FunctionNorm(self.integrator, self.area)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.uh = self.V.function()
        self.uI = self.V.interpolation(self.model.solution)
        self.area = mesh.area()
        self.error.area = self.area 

    def recover_estimate(self):
        if self.V.p > 1:
            raise ValueError('This method only work for p=1!')

        V = self.V
        mesh = V.mesh

        p2c = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = p2c@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = self.uh.grad_value(bc)

        rguh = self.V.function(dim=2)
        rguh[:] = np.asarray(p2c@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.area
        return np.sqrt(e)

    
    def get_left_matrix(self):
        return doperator.stiff_matrix(self.V, self.integrator, self.area)

    def get_right_vector(self):
        return doperator.source_vector(self.model.source, self.V, self.integrator, self.area)

    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')
        #linear_solver(self, self.uh, dirichlet=bc)


    def l2_error(self):
        u = self.model.solution
        uh = self.uh
        return self.error.l2_error(u, uh)

    def L2_error(self):
        u = self.model.solution
        uh = self.uh.value
        mesh = self.mesh
        return self.error.L2_error(u, uh, mesh)

    def H1_semi_error(self):
        gu = self.model.gradient
        guh = self.uh.grad_value
        mesh = self.mesh
        return self.error.L2_error(gu, guh, mesh)

    def recover_error(self, rgh):
        gu = self.model.gradient
        guh = rgh.value
        mesh = self.mesh
        return self.error.L2_error(gu, guh, mesh)
