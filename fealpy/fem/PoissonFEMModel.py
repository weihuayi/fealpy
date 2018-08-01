import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..fem import doperator 

from .integral_alg import IntegralAlg
from ..recovery import FEMFunctionRecoveryAlg

from ..solver import solve
from ..boundarycondition import DirichletBC



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

    def recover_estimate(self,rguh):
        if self.femspace.p > 1:
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
        return doperator.stiff_matrix(self.femspace, self.integrator, self.cellmeasure)

    def get_right_vector(self):
        return doperator.source_vector(self.pde.source, self.femspace, self.integrator, self.cellmeasure)

    def solve(self):
        bc = DirichletBC(self.femspace, self.pde.dirichlet)
        self.A, b= solve(self, self.uh, dirichlet=bc, solver='direct')

    
    def get_l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def get_L2_error(self):
        u = self.pde.solution
        uh = self.uh.value
        L2 = self.integralalg.L2_error(u, uh)
        return L2

    def get_H1_error(self):
        gu = self.pde.gradient
        guh = self.uh.grad_value
        H1 = self.integralalg.L2_error(gu, guh)
        return H1

    def get_recover_error(self,rguh):
        gu = self.pde.gradient
        guh = rguh.value
        mesh = self.mesh
        re = self.integralalg.L2_error(gu, guh, mesh)  
        return re
