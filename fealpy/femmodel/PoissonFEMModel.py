import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..solver.petsc_solver import linear_solver
from ..solver.hofsolver import HOFEMFastSovler
from ..boundarycondition import DirichletBC
from ..femmodel import doperator 
from .integral_alg import IntegralAlg

class PoissonFEMModel(object):
    def __init__(self, pde, mesh,  p, integrator):
        self.femspace = LagrangeFiniteElementSpace(mesh, p) 
        self.mesh = self.femspace.mesh
        self.pde = pde 
        self.uh = self.femspace.function()
        self.uI = self.femspace.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = integrator 
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)

    def recover_estimate(self):
        if self.femspace.p > 1:
            raise ValueError('This method only work for p=1!')

        femspace = self.femspace
        mesh = femspace.mesh

        node2cell = mesh.ds.node_to_cell()
        inv = 1/self.cellmeasure
        asum = node2cell@inv

        bc = np.array([1/3]*3, dtype=np.float)
        guh = self.uh.grad_value(bc)

        rguh = self.femspace.function(dim=mesh.geo_dimension())
        rguh[:] = np.asarray(node2cell@(guh*inv.reshape(-1, 1)))/asum.reshape(-1, 1)

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.cellmeasure
        return np.sqrt(e)

    
    def get_left_matrix(self):
        return doperator.stiff_matrix(self.femspace, self.integrator, self.cellmeasure)

    def get_right_vector(self):
        return doperator.source_vector(self.pde.source, self.femspace, self.integrator, self.cellmeasure)

    def solve(self):
        bc = DirichletBC(self.femspace, self.pde.dirichlet)
        self.A, b= solve(self, self.uh, dirichlet=bc, solver='direct')

    def fast_solve(self):
        bc = DirichletBC(self.femspace, self.pde.dirichlet)
        A = self.get_left_matrix()
        b = self.get_right_vector()
        AD, b = bc.apply(A, b)
        solver = HOFEMFastSovler(AD, self.femspace, self.integrator,
                self.cellmeasure)
        self.uh[:] = solver.solve(b, tol=1e-13)


    def error(self):
        u = self.pde.solution
        uh = self.uh.value
        e0 = self.integralalg.L2_error(u, uh)

        gu = self.pde.gradient
        guh = self.uh.grad_value
        e1 = self.integralalg.L2_error(gu, guh)
        return e0, e1

    def recover_error(self, rgh):
        gu = self.pde.gradient
        guh = rgh.value
        mesh = self.mesh
        return self.integralalg.L2_error(gu, guh, mesh)
