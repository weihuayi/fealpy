import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..solver import solve
from ..boundarycondition import DirichletBC
from  . import doperator 
from .integral_alg import IntegralAlg

class PoissonFEMModel(object):
    def __init__(self, pde, mesh, p, integrator):
        self.femspace = LagrangeFiniteElementSpace(mesh, p) 
        self.mesh = self.femspace.mesh
        self.pde = pde 
        self.uh = self.femspace.function()
        self.uI = self.femspace.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = integrator 
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)

    def get_left_matrix(self):
        return doperator.stiff_matrix(self.femspace, self.integrator, self.cellmeasure)

    def get_right_vector(self):
        return doperator.source_vector(self.pde.source, self.femspace, self.integrator, self.cellmeasure)

    def solve(self):
        bc = DirichletBC(self.femspace, self.pde.dirichlet)
        self.A, b= solve(self, self.uh, dirichlet=bc, solver='direct')
        return self.uh

    def get_L2_error(self):
        u = self.pde.solution
        uh = self.uh.value
        e = self.integralalg.L2_error(u, uh)
        return e 

    def get_H1_error(self):
        gu = self.pde.gradient
        guh = self.uh.grad_value
        e = self.integralalg.L2_error(gu, guh)
        return e
