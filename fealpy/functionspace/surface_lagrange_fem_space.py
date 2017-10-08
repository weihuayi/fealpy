import numpy as np
from numpy.linalg import inv

from .function import FiniteElementFunction
from .lagrange_fem_space import LagrangeFiniteElementSpace2d
from ..common import ranges
from ..quadrature import TriangleQuadrature

class SurfaceTriangleMesh():
    def __init__(self, surf, mesh, p=1, dtype=np.float):
        self.scalarspace = LagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
        self.point, _= surface.project(self.scalarspace.interpolation_points())
        self.dtype=dtype

    def number_of_points(self):
        return self.point.shape[0] 

    def number_of_nodes(self):
        return self.point.shape[0]

    def number_of_edges(self):
        mesh = self.scalarspace.mesh
        return mesh.ds.NE

    def number_of_faces(self):
        mesh = self.scalarspace.mesh
        return mesh.ds.NC

    def number_of_cells(self):
        mesh = self.scalarspace.mesh
        return mesh.ds.NC

    def geom_dimension(self):
        return self.point.shape[1]

    def jacobi(self, bc):
        cell2dof = self.scalarspace.cell_to_dof()
        grad = self.scalarspace.grad_basis(bc)

        # Jacobi 
        J = np.einsum('ij..., ijk->i...k', 
                self.point[cell2dof, :], grad)
        # outward norm at bc
        mesh = self.scalarspace.mesh
        cell = mesh.ds.cell
        F0 = np.einsum('ij..., ij->i...',
                J, mesh.point[cell[:, 1], :] - mesh.point[cell[:, 0], :])
        F1 = np.einsum('ij..., ij->i...',
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
        self.p = p
        self.dtype = dtype
        self.surface = SurfaceTriangleMesh(mesh, surface, p=p, dtype=dtype) 

    def __str__(self):
        return "Lagrange finite element space on surface triangle mesh!"

    def basis(self, bc):
        return self.dsurface.scalarspace.basis(bc)

    def grad_basis(self, bc):
        J, n, grad = self.dsurface.jacobi(bc)
        grad = np.einsum('i...k, ijk->ij...',
                inv(J.transpose(0, 2, 1)), grad)
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
        pass

    def number_of_global_dofs(self):
        pass

    def number_of_local_dofs(self):
        pass

    def interpolation_points(self):
        pass

    def interpolation(self, u, uI):
        pass

    def projection(self, u, up):
        pass

    def array(self):
        pass
