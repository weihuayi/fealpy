
import numpy as np
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace

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
        # the tranpose of the jacobi matrix between S_h and K 
        Jh = mesh.jacobi_matrix(cellidx=cellidx) 

        # the tranpose of the jacobi matrix between S_p and S_h 
        if cellidx is None:
            Jph = np.einsum('ijm, ...ijk->...imk', self.point[cell2dof, :], grad)
        else:
            Jph = np.einsum('ijm, ...ijk->...imk', self.point[cell2dof[cellidx], :], grad)

        # the transpose of the jacobi matrix between S_p and K
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
