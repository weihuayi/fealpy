import numpy as np
from numpy.linalg import inv
from .function import Function

class LinearLagrangeFiniteElementSpace:
    def __init__(self, mesh, surface=None, spacetype='C'):
        """

        Parameters
        ----------

        Returns
        -------

        See Also
        --------
            
        Notes
        -----
        """ 
        self.mesh = mesh 
        self.surface = surface
        
        if spacetype is 'C':
            self.dof = CPLFEMDof2d(mesh, 1) 
            self.TD = 2
            self.GD = mesh.geo_dimension()
        elif spacetype is 'D':
            self.dof = DPLFEMDof2d(mesh, 1) 
            self.TD = 2
            self.GD = mesh.geo_dimension()

    def __str__(self):
        return "Linear Lagrange finite element space!"

    def basis(self, bc):
        """
        Compute all basis function values at a given barrycenter.
        """
        return bc 

    def grad_basis(self, bc, cellidx=None):
        """
        Compute the gradients of all basis functions at a given barrycenter.
        """
        gphi = mesh.grad_lambda()
        return gphi 

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

    def function(self, dim=None):
        f = Function(self, dim=dim)
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
