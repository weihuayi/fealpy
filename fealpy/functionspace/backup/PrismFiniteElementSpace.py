import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .Function import Function
from ..quadrature import FEMeshIntegralAlg
from timeit import default_timer as timer
from .femdof import CPPFEMDof3d

class PrismFiniteElementSpace():

    def __init__(self, mesh, p=1, q=None):
        self.mesh = mesh
        self.p = p

        self.cellmeasure = self.mesh.entity_measure('cell')
        self.dof = CPPFEMDof3d(mesh, p)

        q = p+3 if q is None else q
        self.integralalg = FEMeshIntegralAlg(self.mesh, q, self.cellmeasure)
        self.integrator = self.integralalg.integrator


        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.GD = 3
        self.TD = 3


    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.dpoints

    def cell_to_dof(self):
        return self.dof.cell2dof

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def edge_to_dof(self):
        return self.dof.edge_to_dof()

    def boundary_dof(self, threshold=None):
        return self.dof.boundary_dof(threshold=threshold)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def lagranian_basis(self, bc, TD):
        p = self.p   # the degree of polynomial basis function
        multiIndex = self.dof.multi_index_matrix(TD)
        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def lagranian_grad_basis(self, bc, TD):
        p = self.p   # the degree of polynomial basis function

        multiIndex = self.dof.multi_index_matrix(TD)
        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = multiIndex.shape[0]
        shape = bc.shape[:-1]+(ldof, TD+1)
        gphi = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            gphi[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return gphi

    def basis(self, bcs):
        bc0 = bcs[0] # triangle 
        bc1 = bcs[1] # interval
        phi0 = self.lagranian_basis(bc0, 1)
        phi1 = self.lagranian_basis(bc1, 2)
        phi = np.einsum('ij, kl->ikjl', phi0, phi1)
        shape = phi.shape[0:2] + (1, -1, )
        return phi.reshape(shape) # (NQ0, NQ1, ldof0*ldof1)

    def grad_basis(self, bc, cellidx=None):
        bc0 = bc[0]
        bc1 = bc[1]
        phi0 = self.lagranian_basis(bc0, 1) # (NQ0, ldof0)
        phi1 = self.lagranian_basis(bc1, 2) # (NQ1, ldof1)

        gphi0 = self.lagranian_grad_basis(bc0, 1) # (NQ0, ldof0, 2)
        gphi1 = self.lagranian_grad_basis(bc1, 2) # (NQ1, ldof1, 3)

        NQ0 = bc0.shape[0]
        NQ1 = bc1.shape[0]
        ldof0 = phi0.shape[1]
        ldof1 = phi1.shape[1]

        gphi = np.zeros((NQ0, NQ1, ldof0, ldof1, 3), dtype=self.ftype)

        # gphi0[..., 1]: (NQ0, ldof0)
        # phi1: (NQ1, ldof1)
        # gphi[..., 0]: (NQ0, NQ1, ldof0, ldof1)
        gphi[..., 0] = np.einsum('ij, kl->ikjl', phi0, gphi1[..., 1] - gphi1[..., 0])
        gphi[..., 1] = np.einsum('ij, kl->ikjl', phi0, gphi1[..., 2] - gphi1[..., 0])
        gphi[..., 2] = np.einsum('ij, kl->ikjl', gphi0[..., 1] - gphi0[..., 0], phi1)

        J = inv(self.mesh.jacobi_matrix(bc))

        # J: (NQ0, NQ1, NC, 3, 3)
        # gphi: (NQ0, NQ1, ldof0, ldof1, 3)
        gphi = np.einsum('...imn, ...jln->...ijlm', J, gphi)
        shape = gphi.shape[0:3] + (-1, 3)
        return gphi.reshape(shape)

    def value(self, uh, bcs, cellidx=None):
        phi = self.basis(bcs) # (NQ0, NQ1, ldof0*ldof1)
        cell2dof = self.cell_to_dof() # (NC, ldof0*ldof1)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ki, ki{}->...k{}'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            uh = uh[cell2dof] # (NC, ldof0*ldof1, ...)
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum(s1, phi, uh)
        return val

    def grad_value(self, uh, bcs, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)#(NQ0, NQ1, NC, ldof0*ldof1, GD)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...kim, ki{}->...k{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            uh = uh[cell2dof] # (NC, ldof0, ldof1, ...)
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum(s1, gphi, uh)
        return val

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = u(ipoint)
        return self.function(dim=dim, array=uI)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def stiff_matrix(self, cfun=None):
        p = self.p
        GD = self.geo_dimension()

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        gphi = self.grad_basis(bcs) #(NQ0, NQ1, NC, ldof0*ldof1, GD)
        print(gphi.shape)

        A = np.einsum('..., ...jkm, ...jpm, j->jkp',
                ws, gphi, gphi, self.cellmeasure,
                optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs()

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, cfun=None, barycenter=False):
        p = self.p
        mesh = self.mesh
        cellmeasure = self.cellmeasure

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs)

        M = np.einsum( 'm, mj, mk, i->ijk', ws, dphi, phi, cellmeasure, optimize=True)

        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('ij, k->ijk',  cell2dof, np.ones(ldof))
        J = I.swapaxes(-1, -2)

        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f):
        p = self.p
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        pp = self.mesh.bc_to_point(bcs)
        fval = f(pp) # (NQ0, NQ1, NC)
        phi = self.basis(bcs) # (NQ0, NQ1, ldof)
        # bb: (NC, ldof)
        bb = np.einsum('mn, mni, mnik, i->ik', ws, fval, phi, self.cellmeasure)
        cell2dof = self.cell_to_dof() #(NC, ldof)
        gdof = self.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def set_dirichlet_bc(self, uh, g, is_dirichlet_boundary=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        ipoints = self.interpolation_points()
        isDDof = self.boundary_dof(threshold=is_dirichlet_boundary)
        uh[isDDof] = g(ipoints[isDDof])
        return isDDof

