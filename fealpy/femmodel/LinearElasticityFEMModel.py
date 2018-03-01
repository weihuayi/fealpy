import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat

from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..functionspace.mixed_fem_space import HuZhangFiniteElementSpace
from .integral_alg import IntegralAlg


class LinearElasticityFEMModel:
    def __init__(self, mesh,  model, integrator, p=4):
        self.tensorspace = HuZhangFiniteElementSpace(mesh, p)
        self.vectorspace = VectorLagrangeFiniteElementSpace(mesh, p-1, spacetype='D') 
        self.dim = self.tensorspace.dim
        self.sh = self.tensorspace.function()
        self.uh = self.vectorspace.function()
        self.model = model
        self.integrator = integrator
        self.measure = mesh.entity_measure()
        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.tensorspace.p

        self.tensorspace = HuZhangFiniteElementSpace(mesh, p)
        self.vectorspace = VectorLagrangeFiniteElementSpace(mesh, p-1) 
        self.sh = self.tensorspace.function()
        self.uh = self.vectorspace.function()
        self.integrator = integrator
        self.measure = mesh.entity_measure()
        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)

    def get_left_matrix(self):
        tspace = self.tensorspace
        vspace = self.vectorspace

        bcs, ws = self.integrator.quadpts, self.integrator.weights
        phi = tspace.basis(bcs)
        aphi = self.model.compliance_tensor(phi)
        M = np.einsum('i, ijkmn, ijomn, j->jko', ws, phi, aphi, self.measure)

        tldof = tspace.number_of_local_dofs()
        I = np.einsum('ij, k->ijk', tspace.cell_to_dof(), np.ones(tldof))
        J = I.swapaxes(-1, -2)

        tgdof = tspace.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(tdof, tdof))

        dphi = tspace.div_basis(bcs)
        uphi = vspace.basis(bcs)
        B = np.einsum('i, ikm, ijom, j->jko', ws, uphi, dphi, self.measure)

        vgdof = vspace.number_of_global_dofs()
        vldof = vspace.number_of_local_dofs()
        I = np.einsum('ij, k->ijk', vspace.cell_to_dof(), np.ones(vldof))
        B = csr_matrix((B.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))
        return bmat([[M, B.transpose()], [B, None]]) 

    def get_right_vector(self):
        qf = self.integrator
        vspace = self.vectorspace
        tspace = self.tensorspace

        bcs, ws = qf.quadpts, qf.weights
        pp = vspace.mesh.bc_to_point(bcs)
        fval = self.model.source(pp)
        phi = vspace.basis(bcs)
        bb = np.einsum('i, ikm, ijm, k->kj', ws, fval, phi, self.measure)

        cell2dof = vspace.cell_to_dof()
        vgdof = vspace.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)

        tgdof = tspace.number_of_global_dofs()
        return np.r_['0', np.zeros(tgdof), b]

    def solve(self):
        M, B = self.get_left_matrix()
        b = self.get_right_vector()

        A = bmat([[M, B.transpose()], [B, None]])
