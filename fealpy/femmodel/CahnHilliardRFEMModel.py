import numpy as np
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..femmodel import doperator 
from .integral_alg import IntegralAlg
from .doperator import grad_recovery_matrix, 


class CahnHilliardRFEMModel():
    def __init__(self, pde, n, tau, integrator):
        self.pde = pde 

        self.mesh = pde.space_mesh(n) 
        self.timemesh, self.tau = self.pde.time_mesh(tau)

        self.femspace = LagrangeFiniteElementSpace(mesh, 1) 
        self.uh = self.femspace.function(dim=len(self.timemesh))
        self.uh[:, 0] = self.femspace.interpolation(pde.initdata)

        self.area = mesh.entity_measure('cell')

        self.integrator = integrator 
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)

        self.gradphi = self.mesh.grad_lambda() 
        self.A, self.B = grad_recovery_matrix(self.femspace)
        self.M = mass_matrix(self.femspace, self.integrator, self.area)

    def get_left_matrix(self):
        pass

    def get_right_vector(self):
        pass

    def get_non_linear_vector(self, uh):
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        gradphi = self.femspace.grad_basis()

        uval = uh.value(bcs)
        guval = uh.grad_value(bcs)
        fval = (3*uval[..., np.newaxis]**2 - 1)*guval

        bb = np.einsum('i, ikjm, kjm, k->kj', ws, fval, gradphi, self.area)
        cell2dof = space.cell_to_dof()
        gdof = space.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def solve(self):
        pass
