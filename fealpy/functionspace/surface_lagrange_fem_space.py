import numpy as np
from numpy.linalg import inv

from .function import FiniteElementFunction
from .lagrange_fem_space import LagrangeFiniteElementSpace
from ..common import ranges
from .dof import CPLFEMDof2d, DPLFEMDof2d 

class SurfaceTriangleMesh():
    def __init__(self, mesh, surface, p=1):
        self.mesh = mesh
        self.V = LagrangeFiniteElementSpace(mesh, p)
        self.point, d = surface.project(self.V.interpolation_points())
        self.surface = surface

    def number_of_points(self):
        return self.point.shape[0] 

    def number_of_edges(self):
        return self.mesh.ds.NE

    def number_of_cells(self):
        return self.mesh.ds.NC

    def geom_dimension(self):
        return self.point.shape[1]

    def top_dimension(self):
        return 2

    def jacobi_matrix(self, bc, cellidx=None):
        mesh = self.mesh
        cell = mesh.ds.cell
        cell2dof = self.V.dof.cell2dof

        grad = self.V.grad_basis(bc, cellidx=cellidx)
        Jh = mesh.jacobi_matrix(cellidx=cellidx)
        if cellidx is None:
            Jph = np.einsum('ijm, ...ijk->...imk', self.point[cell2dof, :], grad)
        else:
            Jph = np.einsum('ijm, ...ijk->...imk', self.point[cell2dof[cellidx], :], grad)
        Jp = np.einsum('...ijk, imk->...imj', Jph, Jh)
        grad = np.einsum('ijk, ...imk->...imj', Jh, grad)
        return Jp, grad

    def normal(self, bc, cellidx=None):
        Js, _, ps= self.surface_jacobi_matrix(bc, cellidx=cellidx)
        n = np.cross(Js[..., 0, :], Js[..., 1, :], axis=-1)
        return n, ps

    def surface_jacobi_matrix(self, bc, cellidx=None):
        Jp, grad = self.jacobi_matrix(bc, cellidx=cellidx)
        ps = self.bc_to_point(bc, cellidx=cellidx)
        Jsp = self.surface.jacobi_matrix(ps)
        Js = np.einsum('...ijk, ...imk->...imj', Jsp, Jp)
        return Js, grad, ps


    def bc_to_point(self, bc, cellidx=None):
        basis = self.V.basis(bc)
        cell2dof = self.V.dof.cell2dof
        if cellidx is None:
            bcp = np.einsum('...j, ijk->...ik', basis, self.point[cell2dof, :])
        else:
            bcp = np.einsum('...j, ijk->...ik', basis, self.point[cell2dof[cellidx], :])
        return bcp

    def area(self, integrator):
        mesh = self.mesh
        bcs, ws = integrator.quadpts, integrator.weights 
        Jp, _ = self.jacobi_matrix(bcs)
        n = np.cross(Jp[..., 0, :], Jp[..., 1, :], axis=-1)
        l = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, l)/2.0
        return a


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
        self.mesh = SurfaceTriangleMesh(mesh, surface, p=p) 
        
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

    def grad_value_on_sh(self, uh, bc, cellidx=None):
        pass

    def grad_value_on_surface(self, uh, bc, cellidx=None):
        gphi, ps, n = self.grad_basis_on_surface(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            val = np.einsum('...ijm, ij->...im', gphi, uh[cell2dof])
        else:
            val = np.einsum('...ijm, ij->...im', gphi, uh[cell2dof[cellidx]])
        return val, ps, n

    def hessian_value(self, uh, bc, cellidx=None):
        pass

    def div_value(self, uh, bc, cellidx=None):
        pass
    
    def number_of_global_dofs(self):
        return self.mesh.V.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.mesh.V.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        ipoint, _ = self.mesh.surface.project(ipoint)
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def function(self):
        """
        get a function object in this function space 
        """
        return FiniteElementFunction(self)

    def projection(self, u, up):
        pass

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=np.float)
