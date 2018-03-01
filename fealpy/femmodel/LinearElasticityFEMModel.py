import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..functionspace.mixed_fem_space import HuZhangFiniteElementSpace
from .integral_alg import IntegralAlg
from timeit import default_timer as timer


class LinearElasticityFEMModel:
    def __init__(self, mesh,  model, p, integrator):
        self.mesh = mesh
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

        self.mesh = mesh
        self.tensorspace = HuZhangFiniteElementSpace(mesh, p)
        self.vectorspace = VectorLagrangeFiniteElementSpace(mesh, p-1) 
        self.sh = self.tensorspace.function()
        self.uh = self.vectorspace.function()
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
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(tgdof, tgdof))

        dphi = tspace.div_basis(bcs)
        uphi = vspace.basis(bcs)
        B = np.einsum('i, ikm, ijom, j->jko', ws, uphi, dphi, self.measure)

        vgdof = vspace.number_of_global_dofs()
        vldof = vspace.number_of_local_dofs()
        I = np.einsum('ij, k->ijk', vspace.cell_to_dof(), np.ones(tldof))
        J = np.einsum('ij, k->ikj', tspace.cell_to_dof(), np.ones(vldof))
        B = csr_matrix((B.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))

        return bmat([[M, B.transpose()], [B, None]]) 

    def get_right_vector(self):
        tspace = self.tensorspace
        vspace = self.vectorspace

        bcs, ws = self.integrator.quadpts, self.integrator.weights
        pp = vspace.mesh.bc_to_point(bcs)
        fval = self.model.source(pp)
        phi = vspace.basis(bcs)
        bb = np.einsum('i, ikm, ijm, k->kj', ws, fval, phi, self.measure)

        cell2dof = vspace.cell_to_dof()
        vgdof = vspace.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=vgdof)

        tgdof = tspace.number_of_global_dofs()
        return np.r_[np.zeros(tgdof), -b]

    def solve(self):
        tgdof = self.tensorspace.number_of_global_dofs()
        vgdof = self.vectorspace.number_of_global_dofs()
        gdof = tgdof + vgdof

        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        x = np.r_[self.sh, self.uh]
        end = timer()
        print("Construct linear system time:", end - start)

        isBdDof = np.r_[np.zeros(tgdof, dtype=np.bool), self.vectorspace.boundary_dof()]

        b -= A@x
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd
        b[isBdDof] = x[isBdDof] 

        start = timer()
        x[:] = spsolve(A, b)
        end = timer()
        print("Solve time:", end-start)

        self.sh[:] = x[0:tgdof]
        self.uh[:] = x[tgdof:]

    def error(self):
        mesh = self.mesh

        s = self.model.stress
        sh = self.sh.value 
        e0 = self.integralalg.L2_error(s, sh) 

        ds = self.model.div_stress
        dsh = self.sh.div_value
        e1 = self.integralalg.L2_error(ds, dsh)

        u = self.model.displacement
        uh = self.uh.value
        e2 = self.integralalg.L2_error(u, uh)

        return e0, e1, e2
