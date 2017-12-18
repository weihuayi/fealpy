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
        mesh = self.scalarspace.mesh
        cell = mesh.ds.cell
        cell2dof = self.scalarspace.cell_to_dof()

        grad = self.scalarspace.grad_basis(bc)
        # Jacobi 
        J0 = mesh.point[cell[:, [1, 2]]] - mesh.point[cell[:, [0]]]
        J1 = np.einsum('ij..., ijk->i...k', self.point[cell2dof, :], grad)
        F = np.einsum('ijk, imk->imj', J1, J0)
        return F, J0, J1, grad

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
        triq = TriangleQuadrature(8)
        nQuad = triq.get_number_of_quad_points()
        for i in range(nQuad):
            bc, w = triq.get_gauss_point_and_weight(i)
            F, _, _, _ = self.jacobi(bc)
            n = np.cross(F[:, 0, :], F[:, 1, :], axis=1)
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
        F, J0, J1, grad = self.mesh.jacobi(bc)
        G = np.zeros((len(F), 2, 2), dtype=self.dtype)
        G[:, 0, 0] = np.einsum('ij, ij->i', F[:, 0, :], F[:, 0, :])
        G[:, 0, 1] = np.einsum('ij, ij->i', F[:, 0, :], F[:, 1, :])
        G[:, 1, 0] = G[:, 0, 1]
        G[:, 1, 1] = np.einsum('ij, ij->i', F[:, 1, :], F[:, 1, :])
        G = np.linalg.inv(G)
        grad = np.einsum('ijk, imk->imj', J0, grad)
        grad = np.einsum('ijk, imk->imj', G, grad)
        grad = np.einsum('ijk, imj->imk', F, grad)
        return grad

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        return uh[cell2dof]@phi 
    
    def grad_value(self, uh, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        NC = self.mesh.number_of_cells()
        val = np.zeros((NC, dim), dtype=self.dtype)
        val = np.einsum('ij, ij...->i...', uh[cell2dof], gradphi)
        return val 


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

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def finite_element_function(self):
        return FiniteElementFunction(self)

    def projection(self, u, up):
        pass

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=self.dtype)
