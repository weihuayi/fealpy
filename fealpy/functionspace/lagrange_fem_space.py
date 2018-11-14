import numpy as np

from .function import Function
from .dof import *

class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p 
        if spacetype is 'C':
            if mesh.meshtype is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.dim = 1
            elif mesh.meshtype is 'tri':
                self.dof = CPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshtype is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.dim = 3
        elif spacetype is 'D':
            if mesh.meshtype is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.dim = 1
            elif mesh.meshtype is 'tri':
                self.dof = DPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshtype is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.dim = 3

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def geo_dimension(self):
        return self.dim

    def top_dimension(self):
        return self.dim

    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`         

        Returns
        -------
        phi : numpy.array
            the shape of 'phi' can be `(ldof, )` or `(NQ, ldof)`

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function

        if p == 0 and self.spacetype == 'D':
            if len(bc.shape) == 1:
                return np.ones(1, dtype=self.ftype)
            else:
                return np.ones((bc.shape[0], 1), dtype=self.ftype)

        dim = self.dim 
        multiIndex = self.dof.multiIndex 

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.linspace(0, 1, p, endpoint=False)
        shape = bc.shape[:-1]+(p+1, dim+1)
        A = np.ones(shape, dtype=np.float)
        A[..., 1:, :] = bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(dim+1)
        phi = (p**p)*np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_basis(self, bc, cellidx=None):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`         

        Returns
        -------
        gphi : numpy.array
            the shape of `gphi` can b `(NC, ldof, gdim)' or `(NQ, NC, ldof, gdim)'

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function
        dim = self.dim 

        multiIndex = self.dof.multiIndex 

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)

        t = np.linspace(0, 1, p, endpoint=False)
        shape = bc.shape[:-1]+(p+1, dim+1)
        A = np.ones(shape, dtype=np.float)
        A[..., 1:, :] = bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = 1
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=np.float)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(dim+1)]
        M = F[..., multiIndex, range(dim+1)]
        ldof = self.number_of_local_dofs()
        shape = bc.shape[:-1]+(ldof, dim+1)
        R = np.zeros(shape, dtype=np.float)
        for i in range(dim+1):
            idx = list(range(dim+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        pp = p**p
        Dlambda = self.mesh.grad_lambda()
        if cellidx is None:
            gphi = np.einsum('...ij, kjm->...kim', pp*R, Dlambda)
        else:
            gphi = np.einsum('...ij, kjm->...kim', pp*R, Dlambda[cellidx, :, :])
        return gphi 

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...j, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof]) 
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]]) 
        return val 

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        dim = len(uh.shape)
        gdim = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == gdim):
            val = self.grad_value(uh, bc, cellidx=cellidx)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")


    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = Function(self, dim=dim)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def function(self, dim=None):
        f = Function(self, dim=dim)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

class VectorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.dim = self.scalarspace.dim

    def __str__(self):
        return "Vector Lagrange finite element space!"

    def geo_dimension(self):
        return self.dim

    def vector_dim(self):
        return self.dim

    def cell_to_dof(self):
        dim = self.dim
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = dim*cell2dof + np.arange(dim)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof_flag(self):
        dim = self.dim
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, dim)

    def number_of_global_dofs(self):
        return self.dim*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dim*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        dim = self.dim
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(self.dim))
        shape += [-1, dim] 
        phi = phi.reshape(shape)
        return phi

    def div_basis(self, bcs, cellidx=None):
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1]
        return gphi.reshape(shape)

    def value(self, uh, bcs, cellidx=None):
        phi = self.basis(bcs)
        cell2dof = self.cell_to_dof()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...jm, ij->...im',  phi, uh) 
        return val 

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)

    def interpolation(self, u):
        dim = self.dim
        c2d = self.dof.cell2dof
        ldof = self.dof.number_of_local_dofs()
        cell2dof = self.cell_to_dof().reshape(-1, ldof, dim)
        p = self.dof.interpolation_points()[c2d]
        uI = Function(self)
        uI[cell2dof] = u(p)
        return uI

class SymmetricTensorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.dim = self.scalarspace.dim

        if self.dim == 2:
            self.T = np.array([[(1, 0), (0, 0)], [(0, 1), (1, 0)], [(0, 0), (0, 1)]])
        elif self.dim == 3:
            self.T = np.array([
                [(1, 0, 0), (0, 0, 0), (0, 0, 0)], 
                [(0, 1, 0), (1, 0, 0), (0, 0, 0)],
                [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
                [(0, 0, 0), (0, 1, 0), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
                [(0, 0, 0), (0, 0, 0), (0, 0, 1)]])

    def __str__(self):
        return " Symmetric Tensor Lagrange finite element space!"

    def geom_dim(self):
        return self.dim

    def tensor_dim(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def cell_to_dof(self):
        tdim = self.tensor_dim()
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = tdim*cell2dof + np.arange(tdim)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof(self):
        tdim = self.tensor_dim()
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, tdim)

    def number_of_global_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        dim = self.dim
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mno->...jmno', phi0, self.T)
        shape += [-1, dim, dim]
        return phi.reshape(shape)

    def div_basis(self, bcs, cellidx=None):
        dim = self.dim
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1, dim]
        return gphi.reshape(shape)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        uh0 = uh.reshape(-1, self.dim)
        if cellidx is None:
            uh0 = uh0[cell2dof].reshape(-1)
        else:
            uh0 = uh0[cell2dof[cellidx]].reshape(-1)
        val = np.einsum('...jm, ij->...im',  phi, uh0) 
        return val 

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)

    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        uI = Function(self)
        uI[:] = u(ipoint).flat[:]
        return uI
