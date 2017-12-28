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
        Jh = mesh.point[cell[:, [1, 2]]] - mesh.point[cell[:, [0]]]
        Jph = np.einsum('ij..., ijk->i...k', self.point[cell2dof, :], grad)
        Jp = np.einsum('ijk, imk->imj', Jph, Jh)
        grad = np.einsum('ijk, imk->imj', Jh, grad)
        return Jp, grad

    def normal(self, bc):
        Js, _, ps= self.surface_jacobi(bc)
        n = np.cross(Js[:, 0, :], Js[:, 1, :], axis=1)
        return n, ps

    def surface_jacobi(self, bc):
        Jp, grad = self.jacobi(bc)
        ps = self.bc_to_point(bc)
        Jsp = self.surface.jacobi(ps)
        Js = np.einsum('ijk, imk->imj', Jsp, Jp)
        return Js, grad, ps


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
        qf = TriangleQuadrature(8)
        nQuad = qf.get_number_of_quad_points()
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            Jp, _ = self.jacobi(bc)
            n = np.cross(Jp[:, 0, :], Jp[:, 1, :], axis=1)
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
        self.cell2dof = self.cell_to_dof()
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
        Jp, grad = self.mesh.jacobi(bc)
        Gp = np.zeros((len(Jp), 2, 2), dtype=self.dtype)
        Gp[:, 0, 0] = np.einsum('ij, ij->i', Jp[:, 0, :], Jp[:, 0, :])
        Gp[:, 0, 1] = np.einsum('ij, ij->i', Jp[:, 0, :], Jp[:, 1, :])
        Gp[:, 1, 0] = Gp[:, 0, 1]
        Gp[:, 1, 1] = np.einsum('ij, ij->i', Jp[:, 1, :], Jp[:, 1, :])
        Gp = np.linalg.inv(Gp)
        grad = np.einsum('ijk, imk->imj', Gp, grad)
        grad = np.einsum('ijk, imj->imk', Jp, grad)
        return grad

    def grad_basis_on_surface(self, bc):
        Js, grad, ps = self.mesh.surface_jacobi(bc)
        Gs = np.zeros((len(Js), 2, 2), dtype=self.dtype)
        Gs[:, 0, 0] = np.einsum('ij, ij->i', Js[:, 0, :], Js[:, 0, :])
        Gs[:, 0, 1] = np.einsum('ij, ij->i', Js[:, 0, :], Js[:, 1, :])
        Gs[:, 1, 0] = Gs[:, 0, 1]
        Gs[:, 1, 1] = np.einsum('ij, ij->i', Js[:, 1, :], Js[:, 1, :])
        Gs = np.linalg.inv(Gs)
        grad = np.einsum('ijk, imk->imj', Gs, grad)
        grad = np.einsum('ijk, imj->imk', Js, grad)
        n = np.cross(Js[:, 0, :], Js[:, 1, :], axis=1)
        return grad, ps, n

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
        grad, ps = self.grad_basis(bc)
        cell2dof = self.cell2dof
        val = np.einsum('ij, ij...->i...', uh[cell2dof], gradphi)
        return val 

    def grad_value_on_sh(self, uh, bc):
        pass

    def grad_value_on_surface(self, uh, bc):
        mesh = self.mesh
        cell2dof = self.cell2dof
        grad, ps, n = self.grad_basis_on_surface(bc)
        val = np.einsum('ij, ij...->i...', uh[cell2dof], grad)
        return val, ps, n


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
