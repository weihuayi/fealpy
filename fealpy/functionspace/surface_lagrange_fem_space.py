import numpy as np
from numpy.linalg import inv

from .function import FiniteElementFunction
from .lagrange_fem_space import LagrangeFiniteElementSpace2d
from ..common import ranges
from ..quadrature import TriangleQuadrature

class SurfaceTriangleMesh():
    def __init__(self, mesh, surface, p=1, dtype=np.float):
        self.scalarspace = LagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
        self.point, d = surface.project(self.scalarspace.interpolation_points())
        self.surface = surface

        self.dtype=dtype

    def number_of_points(self):
        return self.point.shape[0] 

    def number_of_edges(self):
        mesh = self.scalarspace.mesh
        return mesh.ds.NE

    def number_of_cells(self):
        mesh = self.scalarspace.mesh
        return mesh.ds.NC

    def geom_dimension(self):
        return self.point.shape[1]

    def toplogy_dimension(self):
        return 2

    def jacobi(self, bc):
        cell2dof = self.scalarspace.cell_to_dof()
        grad = self.scalarspace.grad_basis(bc)

        # Jacobi 
        J = np.einsum('ij..., ijk->i...k', 
                self.point[cell2dof, :], grad)
        # outward norm at bc
        mesh = self.scalarspace.mesh
        cell = mesh.ds.cell
        F0 = np.einsum('i...j, ij->i...',
                J, mesh.point[cell[:, 1], :] - mesh.point[cell[:, 0], :])
        F1 = np.einsum('i...j, ij->i...',
                J, mesh.point[cell[:, 2], :] - mesh.point[cell[:, 0], :])
        return J, F0, F1, grad

    def bc_to_point(self, bc):
        basis = self.scalarspace.basis(bc)
        cell2dof = self.scalarspace.cell_to_dof()
        bcp = np.einsum('j, ijk->ik', 
                basis, self.point[cell2dof, :])
        return bcp

    def area(self):
        mesh = self.scalarspace.mesh
        NC = mesh.number_of_cells()
        a = np.zeros(NC, dtype=self.dtype)
        p = self.scalarspace.p
        triq = TriangleQuadrature(p)
        nQuad = triq.get_number_of_quad_points()
        for i in range(nQuad):
            bc, w = triq.get_gauss_point_and_weight(i)
            _, F0, F1, _ = self.jacobi(bc)
            n = np.cross(F0, F1, axis=1)
            a += np.sqrt(np.sum(n**2, axis=1))*w
        return a/2.0


class SurfaceLagrangeFiniteElementSpace:
    def __init__(self, mesh, surface, p=1, dtype=np.float):
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
        self.mesh = SurfaceTriangleMesh(mesh, surface, p=p, dtype=dtype) 
        self.dtype = dtype

    def __str__(self):
        return "Lagrange finite element space on surface triangle mesh!"

    def basis(self, bc):
        """
        Compute all basis function values at a given barrycenter.
        """
        return self.mesh.scalarspace.basis(bc)

    def grad_basis(self, bc):
        """
        Compute the gradients of all basis functions at a given barrycenter.
        """
        J, F0, F1, grad = self.dsurface.jacobi(bc)
        n = np.cross(F0, F1, axis=1)
        n /= np.sqrt(np.sum(n**2, axis=1, keepdims=True))
        grad = np.einsum('i...k, ijk->ij...', inv(J), grad)
        pgrad = np.einsum('ik, ijk->ij', n, grad)
        grad -= np.einsum('ij, ik->ijk', pgrad, n) 
        return grad

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass
    
    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass
    
    def cell_to_dof(self):
        return self.mesh.scalarspace.cell_to_dof()

    def number_of_global_dofs(self):
        return self.mesh.scalarspace.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.mesh.scalarspace.number_of_local_dofs()

    def interpolation_points(self):
        return self.mesh.point

    def interpolation(self, u, uI):
        pass

    def projection(self, u, up):
        pass

    def array(self):
        pass
