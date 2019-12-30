import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .function import Function
from ..quadrature import FEMeshIntegralAlg
from timeit import default_timer as timer
from .femdof import CPPFEMDof3d

class PrismFiniteElementSpace():

    def __init__(self, mesh, p=1, q=None):
        self.mesh = mesh
        self.p = p

        self.dof = CPPFEMDof3d(mesh, p)

        q = p+3 if q is None else q
        self.integralalg = FEMeshIntegralAlg(self.mesh, q)
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

    def boundary_dof(self):
        return self.dof.boundary_dof()

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

    def basis(self, bc):
        bc0 = bc[0] # triangle 
        bc1 = bc[1] # interval
        phi0 = self.lagranian_basis(bc0, 1)
        phi1 = self.lagranian_basis(bc1, 2)
        phi = np.einsum('ij, kl->ikjl', phi0, phi1)
        return phi

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
        return gphi

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
