import numpy as np
from numpy.linalg import inv

from .function import Function
from ..common import ranges
from .dof import CPLFEMDof2d, DPLFEMDof2d 
from ..mesh.SurfaceTriangleMesh import SurfaceTriangleMesh 

class SurfaceLagrangeFiniteElementSpace:
    def __init__(self, mesh, surface, p=1, p0=None, spacetype='C'):
        """
        Initial a object of SurfaceLagrangeFiniteElementSpace. 

        Parameters
        ----------
        self : 
            Object 
        mesh : 
            This is a mesh object
        surface : 
            The continuous surface which was represented as a level set
            function.
        p : int
            The degree of the Lagrangian space 
        p0 : int
            The degree of the surface mesh

        Returns
        -------

        See Also
        --------
            
        Notes
        -----
        """ 
        if p0 is None:
            p0 = p

        self.p = p
        self.mesh = SurfaceTriangleMesh(mesh, surface, p=p0) 
        self.surface = surface
        
        if p0 == p:
            self.dof = self.mesh.V.dof
            self.dim = 2
        else:
            if spacetype is 'C':
                self.dof = CPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif spacetype is 'D':
                self.dof = DPLFEMDof2d(mesh, p) 
                self.dim = 2

    def __str__(self):
        return "Lagrange finite element space on surface triangle mesh!"

    def basis(self, bc):
        """
        Compute all basis function values at a given barrycenter.
        """
        return self.mesh.V.basis(bc)

    def grad_basis(self, bc, cellidx=None):
        """
        Compute the gradients of all basis functions at a given barrycenter.
        """
        Jp, grad = self.mesh.jacobi_matrix(bc, cellidx=cellidx)
        Gp = np.einsum('...ijk, ...imk->...ijm', Jp, Jp)
        Gp = np.linalg.inv(Gp)
        grad = np.einsum('...ijk, ...imk->...imj', Gp, grad)
        grad = np.einsum('...ijk, ...imj->...imk', Jp, grad)
        return grad

    def grad_basis_on_surface(self, bc, cellidx=None):
        Js, grad, ps = self.mesh.surface_jacobi_matrix(bc, cellidx=cellidx)
        Gs = np.einsum('...ijk, ...imk->...ijm', Js, Js)
        Gs = np.linalg.inv(Gs)
        grad = np.einsum('...ijk, ...imk->...imj', Gs, grad)
        grad = np.einsum('...ijk, ...imj->...imk', Js, grad)
        n = np.cross(Js[..., 0, :], Js[..., 1, :], axis=-1)
        return grad, ps, n

    def hessian_basis(self, bc, cellidx=None):
        pass

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
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
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def grad_value_on_surface(self, uh, bc, cellidx=None):
        gphi, ps, n = self.grad_basis_on_surface(bc, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val, ps, n

    def hessian_value(self, uh, bc, cellidx=None):
        pass

    def div_value(self, uh, bc, cellidx=None):
        pass
    
    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        ipoint, _ = self.surface.project(self.dof.interpolation_points())
        return ipoint
    
    def cell_to_dof(self):
        return self.dof.cell2dof

    def interpolation(self, u, dim=None):
        ipoint = self.interpolation_points()
        uI = Function(self, dim=dim)
        uI[:] = u(ipoint)
        return uI

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def projection(self, u, up):
        pass

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

