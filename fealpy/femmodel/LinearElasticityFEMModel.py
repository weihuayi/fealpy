import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat
from scipy.sparse.linalg import cg, inv, dsolve, spsolve, lsmr

from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..functionspace.mixed_fem_space import HuZhangFiniteElementSpace
from ..solver.petsc_solver import minres
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
        self.sI = self.tensorspace.interpolation(self.model.stress)
        self.integrator = integrator
        self.measure = mesh.entity_measure()
        print(np.max(self.measure))
        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)
        self.count = 0

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.tensorspace.p

        self.count += 1
        self.mesh = mesh
        self.tensorspace = HuZhangFiniteElementSpace(mesh, p)
        self.vectorspace = VectorLagrangeFiniteElementSpace(mesh, p-1) 
        self.sh = self.tensorspace.function()
        self.sI = self.tensorspace.interpolation(self.model.stress)
        self.uh = self.vectorspace.function()
        self.measure = mesh.entity_measure()
        print(np.max(self.measure))
        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)

    def get_left_matrix(self):
        tspace = self.tensorspace
        vspace = self.vectorspace

        bcs, ws = self.integrator.quadpts, self.integrator.weights
        phi = tspace.basis(bcs)
        aphi = self.model.compliance_tensor(phi)
        M = np.einsum('i, ijkmn, ijomn, j->jko', ws, aphi, phi, self.measure)

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

    def average_trace(self):
        dim = self.dim
        sh = self.sh
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        f = lambda x: sh.value(x).trace(axis1=-2, axis2=-1)
        val = self.integralalg.integral(f)/(dim*np.sum(self.measure))
        return val

        
    def write_system(self, M, B, b, sparse=True):
        fM = open('M{}.dat'.format(self.count), 'ab')
        fB = open('B{}.dat'.format(self.count), 'ab')
        fb = open('b{}.dat'.format(self.count), 'ab')

        np.savetxt(fM, np.array([M.shape]), fmt='%d')
        if sparse is True:
            np.savetxt(fM, M.indptr, fmt='%d')
            np.savetxt(fM, np.array([M.indices, M.data]).T, fmt=['%d', '%.18e'])
        else:
            nnz = M.nnz
            np.savetxt(fM, [nnz], fmt='%d')
            T = M.tocoo()
            np.savetxt(fM, np.array([T.row, T.col, T.data]).T, fmt=['%d', '%d', '%.18e'])
        fM.close()

        np.savetxt(fB, np.array([B.shape]), fmt='%d')
        if sparse is True:
            np.savetxt(fB, B.indptr, fmt='%d')
            np.savetxt(fB, np.array([B.indices, B.data]).T, fmt=['%d', '%.18e'])
        else:
            nnz = B.nnz
            np.savetxt(fB, [nnz], fmt='%d')
            T = B.tocoo()
            np.savetxt(fB, np.array([T.row, T.col, T.data]).T, fmt=['%d', '%d', '%.18e'])
        fB.close()

        np.savetxt(fb, [len(b)], fmt='%d')
        np.savetxt(fb, b)
        fb.close()


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

#        c2d = self.tensorspace.cell_to_dof()
#        print(c2d.reshape(1, 10, 3))
#        MM = M.todense()
#
#        f = open('M.txt', 'ab')
#        np.savetxt(f, MM, fmt='%.16f')
#        f.close()
#
#        f = open('B.txt', 'ab')
#        np.savetxt(f, B.todense(), fmt='%.16f')
#        f.close()

        #self.write_system(M, B, b, sparse=False)

        #isBdDof[0:3] = True
        #isBdDof[tgdof:] = self.vectorspace.boundary_dof()
        #x = np.zeros(gdof, dtype=np.float)
        #b -= A@x
        #bdIdx = np.zeros(gdof, dtype=np.int)
        #bdIdx[isBdDof] = 1
        #Tbd = spdiags(bdIdx, 0, gdof, gdof)
        #T = spdiags(1-bdIdx, 0, gdof, gdof)
        #A = T@A@T + Tbd
        #b[isBdDof] = x[isBdDof] 
        #x[:] = spsolve(A, b)


        #x, info =  minres(A, b, tol=1e-8, maxiter=1000, show=True)
        #print('info:', info)

        #x = np.r_[self.sh, self.uh]
        #minres(A, b, x)

        #x, istop = lsmr(A, b)[:2]
        #print(istop)

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
