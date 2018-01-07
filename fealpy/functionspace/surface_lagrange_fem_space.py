import numpy as np
from numpy.linalg import inv

from .function import FiniteElementFunction
from .lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..common import ranges
from ..quadrature import TriangleQuadrature

class SurfaceTriangleMesh():
    def __init__(self, mesh, surface, p=1):
        self.mesh = mesh
        self.V = VectorLagrangeFiniteElementSpace(mesh, p)
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

    def jacobi(self, bc, cellidx=None):
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
        Js, _, ps= self.surface_jacobi(bc, cellidx=cellidx)
        n = np.cross(Js[..., 0, :], Js[..., 1, :], axis=-1)
        return n, ps

    def surface_jacobi(self, bc, cellidx=None):
        Jp, grad = self.jacobi(bc, cellidx=cellidx)
        ps = self.bc_to_point(bc, cellidx=cellidx)
        Jsp = self.surface.jacobi(ps)
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

    def area(self, order=3):
        mesh = self.mesh
        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights 
        Jp = self.jacobi(bcs)
        n = np.cross(Jp[..., 0, :], Jp[..., 1, :], axis=-1)
        l = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, l)/2.0
        return a


class SurfaceLagrangeFiniteElementSpace:
    def __init__(self, mesh, surface, p=1):
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

        Returns
        -------

        See Also
        --------
            
        Notes
        -----
        """ 
        self.p = p
        self.mesh = SurfaceTriangleMesh(mesh, surface, p=p) 
        self.dof = self.mesh.V.dof

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
        Jp, grad = self.mesh.jacobi(bc, cellidx=cellidx)
        Gp = np.einsum('...ijk, ...imk->...ijm', Jp, Jp)
        Gp = np.linalg.inv(Gp)
        grad = np.einsum('...ijk, ...imk->...imj', Gp, grad)
        grad = np.einsum('...ijk, ...imj->...imk', Jp, grad)
        return grad

    def grad_basis_on_surface(self, bc, cellidx=None):
        Js, grad, ps = self.mesh.surface_jacobi(bc, cellidx=cellidx)
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
            val = np.einsum('...j, ijm->...im', phi, uh[cell2dof])
        else:
            val = np.einsum('...j, ijm->...im', phi, uh[cell2dof[cellidx]])
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

    def grad_value_on_surface(self, uh, bc):
        gphi, ps, n = self.grad_basis_on_surface(bc, cellidx=cellidx)
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
    
    def number_of_global_dofs(self):
        return self.mesh.V.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.mesh.V.number_of_local_dofs()

    def interpolation_points(self):
        return self.mesh.point

    def interpolation(self, u):
        ipoint = self.interpolation_points()
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
        return np.zeros((gdof,), dtype=self.dtype)
