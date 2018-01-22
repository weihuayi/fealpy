import numpy as np

from .function import FiniteElementFunction
from ..common import ranges
from .dof import *

class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p 
        if spacetype is 'C':
            if mesh.meshType is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.dim = 1
            elif mesh.meshType is 'tri':
                self.dof = CPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshType is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.dim = 3
        elif spacetype is 'D':
            if mesh.meshType is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.dim = 1
            elif mesh.meshType is 'tri':
                self.dof = DPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshType is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.dim = 3

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(dim+1,)` or `(nb, dim+1)`         

        Returns
        -------
        phi : numpy.array

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
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        phi = (p**p)*np.prod(A[..., multiIndex, [0, 1, 2]], axis=-1)
        return phi

    def grad_basis(self, bc, cellidx=None):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(dim+1,)` or `(nb, dim+1)`         

        Returns
        -------
        phi : numpy.array

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
        F = np.zeros(shape, dtype=np.float)
        A[..., 1:, :] = bc[..., np.newaxis, :] - t.reshape(-1, 1)
        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = 1
        np.cumprod(FF, axis=-2, out=FF)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)
        B = np.cumprod(A, axis=-2)
        B[..., 1:, :] *= P.reshape(-1, 1)

        Q = B[..., multiIndex, [0,1,2]]
        M = F[..., multiIndex, [0,1,2]]
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
        if cellidx is None:
            val = np.einsum('...j, ij->...i', phi, uh[cell2dof]) 
        else:
            val = np.einsum('...j, ij->...i', phi, uh[cell2dof[cellidx]]) 
        return val 

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            val = np.einsum('...ijm, ij->...im', gphi, uh[cell2dof])
        else:
            val = np.einsum('...ijm, ij->...im', gphi, uh[cell2dof[cellidx]])
        return val

    def hessian_value(self, uh, bc, cellidx=None):
        pass

    def div_value(self, uh, bc, cellidx=None):
        pass


    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=np.float)

class VectorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, vectordim=None, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.dof = self.scalarspace.dof 
        if vectordim is None:
            self.vectordim = mesh.geom_dimension()
        else:
            self.vectordim = vectordim 

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc):
        return self.scalarspace.basis(bc)

    def grad_basis(self, bc, cellidx=None):
        return self.scalarspace.grad_basis(bc, cellidx=cellidx)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            val = np.einsum('...j, ijm->...im', phi, uh[cell2dof])
        else:
            val = np.einsum('...j, ijm->...im', phi, uh[cell2dof[cellidx]])
        return val 

    def grad_value(self, uh, bc, cellidx=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            val = np.einsum('...ijm, ijk->...ikm', gphi, uh[cell2dof])
        else:
            val = np.einsum('...ijm, ijk->...ikm', gphi, uh[cell2dof[cellidx]])
        return val


    def div_value(self, uh, bc, cellidx=None):
        val = self.grad_value(uh, bc, cellidx=cellidx)
        return np.sum(np.diagonal(val, axis1=-2, axis2=-1), axis=-1) 

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof, self.vectordim), dtype=np.float)

class TensorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p, tensorshape, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.dof = self.scalarspace.dof 
        if type(tensorshape) is int:
            self.tensorshape = (tensorshape, )
        elif isinstance(tensorshape, tuple):
            self.tensorshape = tensorshape
        else:
            raise ValueError("the type of `tensorshape` must be `int` or `tuple`!")

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc):
        return self.scalarspace.basis(bc)

    def grad_basis(self, bc, cellidx=None):
        return self.scalarspace.grad_basis(bc, cellidx=cellidx)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        s0 = 'abcdefg'
        s1 = '...j, ij{}->...i{}'.format(s0[:len(self.tensorshape)], s0[:len(self.tensorshape)])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val 

    def grad_value(self, uh, bc, cellidx=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:len(self.tensorshape)], s0[:len(self.tensorshape)])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val


    def div_value(self, uh, bc, cellidx=None):
        val = self.grad_value(uh, bc, cellidx=cellidx)
        return np.sum(np.diagonal(val, axis1=-2, axis2=-1), axis=-1) 

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,)+self.tensorshape, dtype=np.float)
