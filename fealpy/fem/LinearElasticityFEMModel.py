import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg, inv, dsolve, spsolve, gmres, LinearOperator, spsolve_triangular
import pyamg

from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.mixed_fem_space import HuZhangFiniteElementSpace
from .integral_alg import IntegralAlg
from .doperator import stiff_matrix
from timeit import default_timer as timer
import cProfile

class LinearElasticityFEMModel:
    def __init__(self, mesh,  model, p, integrator):
        self.mesh = mesh
        self.tensorspace = HuZhangFiniteElementSpace(mesh, p)
        self.vectorspace = VectorLagrangeFiniteElementSpace(mesh, p-1, spacetype='D') 
        self.cspace = LagrangeFiniteElementSpace(mesh, 1) # linear space 
        self.dim = self.tensorspace.dim
        self.sh = self.tensorspace.function()
        self.uh = self.vectorspace.function()
        self.model = model
        self.sI = self.tensorspace.interpolation(self.model.stress)
        self.integrator = integrator
        self.measure = mesh.entity_measure()
        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)
        self.count = 0

    def precondieitoner(self):

        tspace = self.tensorspace
        vspace = self.vectorspace
        tgdof = tspace.number_of_global_dofs()

        gdim = tspace.geo_dimension()

        bcs, ws = self.integrator.quadpts, self.integrator.weights
        phi = tspace.basis(bcs)

        # construct diag matrix D
        if gdim == 2:
            d = np.array([1, 1, 2])
        elif gdim == 3:
            d = np.array([1, 1, 1, 2, 2, 2])
        D = np.einsum('i, ijkm, m, ijkm, j->jk', ws, phi, d, phi, self.measure) 

        tcell2dof = tspace.cell_to_dof()
        self.D = np.bincount(tcell2dof.flat, weights=D.flat, minlength=tgdof)

        # construct amg solver 
        A = stiff_matrix(self.cspace, self.integrator, self.measure)
        isBdDof = self.cspace.boundary_dof()
        bdIdx = np.zeros((A.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(A) # 这里要求必须有网格内部节点 

        # Get interpolation matrix 
        NC = self.mesh.number_of_cells()
        bc = self.vectorspace.dof.multiIndex/self.vectorspace.p
        val = np.tile(bc, (NC, 1))
        c2d0 = self.vectorspace.dof.cell2dof
        c2d1 = self.cspace.cell_to_dof()

        gdim = self.tensorspace.geo_dimension()
        I = np.einsum('ij, k->ijk', c2d0, np.ones(gdim+1))
        J = np.einsum('ik, j->ijk', c2d1, np.ones(len(bc)))
        cgdof = self.cspace.number_of_global_dofs()
        fgdof = self.vectorspace.number_of_global_dofs()/self.mesh.geo_dimension()
        self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(fgdof, cgdof))

    def get_left_matrix(self):
        tspace = self.tensorspace
        vspace = self.vectorspace

        gdim = tspace.geo_dimension()

        bcs, ws = self.integrator.quadpts, self.integrator.weights
        phi = tspace.basis(bcs)
        aphi = self.model.compliance_tensor(phi)
        
        if gdim == 2:
            d = np.array([1, 1, 2])
        elif gdim == 3:
            d = np.array([1, 1, 1, 2, 2, 2])

        M = np.einsum('i, ijkm, m, ijom, j->jko', ws, aphi, d, phi, self.measure)

        tcell2dof = tspace.cell_to_dof()
        tldof = tspace.number_of_local_dofs()
        I = np.einsum('ij, k->ijk', tcell2dof, np.ones(tldof))
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
        return  M, B

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
        return  -b

    def solve(self):
        tgdof = self.tensorspace.number_of_global_dofs()
        vgdof = self.vectorspace.number_of_global_dofs()
        gdof = tgdof + vgdof

        start = timer()
        M, B = self.get_left_matrix()
        b = self.get_right_vector()
        A = bmat([[M, B.transpose()], [B, None]]).tocsr()
        bb = np.r_[np.zeros(tgdof), b]
        end = timer()
        print("Construct linear system time:", end - start)

        start = timer()
        x = spsolve(A, bb)
        end = timer()
        print("Solve time:", end-start)
        self.sh[:] = x[0:tgdof]
        self.uh[:] = x[tgdof:]

    def fast_solve(self):

        self.precondieitoner()

        tgdof = self.tensorspace.number_of_global_dofs()
        vgdof = self.vectorspace.number_of_global_dofs()
        gdof = tgdof + vgdof

        start = timer()
        print("Construting linear system ......!")
        self.M, self.B = self.get_left_matrix()
        S = self.B@spdiags(1/self.D, 0, tgdof, tgdof)@self.B.transpose()
        self.SL = tril(S).tocsc()
        self.SU = triu(S, k=1).tocsr()

        self.SUT = self.SL.transpose().tocsr()
        self.SLT = self.SU.transpose().tocsr()

        b = self.get_right_vector()

        AA = bmat([[self.M, self.B.transpose()], [self.B, None]]).tocsr()
        bb = np.r_[np.zeros(tgdof), b]
        end = timer()
        print("Construct linear system time:", end - start)


        start = timer()
        P = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        x, exitCode = gmres(AA, bb, M=P, tol=1e-8)
        print(exitCode)
        end = timer()

        print("Solve time:", end-start)
        self.sh[:] = x[0:tgdof]
        self.uh[:] = x[tgdof:]

    def linear_operator(self, r):
        tgdof = self.tensorspace.number_of_global_dofs()
        vgdof = self.vectorspace.number_of_global_dofs()
        gdof = tgdof + vgdof
        gdim = self.tensorspace.geo_dimension()

        r0 = r[0:tgdof]
        r1 = r[tgdof:]

        u0 = r0/self.D
        u1 = np.zeros(vgdof, dtype=np.float)
        r2 = r1 - self.B@u0

        for i in range(3):
            u1[:] = spsolve(self.SL, r2 - self.SU@u1, permc_spec="NATURAL") 

        r3 = r2 - (self.SL@u1 + self.SU@u1)
        for i in range(gdim):
            u1[i::gdim] = self.PI@self.ml.solve(self.PI.transpose()@r3[i::gdim], tol=1e-8, accel='cg')

        for i in range(3):
            u1[:] = spsolve(self.SUT, r2 - self.SLT@u1, permc_spec="NATURAL")

        return np.r_[u0 + self.B.transpose()@u1, -u1]

    def error(self):

        dim = self.dim
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

        sI = self.sI.value 
        e3 = self.integralalg.L2_error(s, sI)

        dsI = self.sI.div_value
        e4 = self.integralalg.L2_error(ds, dsI)

        return e0, e1, e2, e3, e4
