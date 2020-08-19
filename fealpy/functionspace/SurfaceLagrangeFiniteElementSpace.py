import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csr_matrix, spdiags

from ..mesh import SurfaceTriangleMesh
from ..quadrature.FEMeshIntegralAlg import FEMeshIntegralAlg

from .femdof import CPLFEMDof2d, DPLFEMDof2d
from .Function import Function


class SurfaceLagrangeFiniteElementSpace:
    def __init__(self, mesh, surface, p=1, p0=None, q=None, spacetype='C', scale=None):
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

        self.scale = scale
        self.p = p

        self.mesh = SurfaceTriangleMesh(mesh, surface, p=p0, scale=self.scale)
        self.surface = surface

        self.cellmeasure = self.mesh.entity_measure('cell')

        if p0 == p:
            self.dof = self.mesh.space.dof
            self.dim = 2
        else:
            if spacetype == 'C':
                self.dof = CPLFEMDof2d(mesh, p)
                self.dim = 2
            elif spacetype == 'D':
                self.dof = DPLFEMDof2d(mesh, p)
                self.dim = 2

        q = q if q is not None else p+3
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def __str__(self):
        return "Lagrange finite element space on surface triangle mesh!"

    def grad_recovery(self, uh, method='area_harmonic'):
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        ldof = self.number_of_local_dofs()
        p = self.p
        bc = self.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = self.function(dim=3)
        if method == 'area_harmonic':
            measure = 1/self.cellmeasure
            ws = np.einsum('i, j->ij', measure, np.ones(ldof))
            deg = np.bincount(cell2dof.flat, weights = ws.flat, minlength=gdof)
            guh = np.einsum('ij..., i->ij...', guh, measure)
            np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            rguh /= deg.reshape(-1, 1)
        else:
            rguh = None
        return rguh

    def stiff_matrix(self):
        p = self.p
        GD = self.mesh.geo_dimension()

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        gphi = self.grad_basis(bcs)

        # Compute the element sitffness matrix
        A = np.einsum('i, ijkm, ijpm, j->jkp', ws, gphi, gphi, self.cellmeasure, optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        # Construct the stiffness matrix
        gdof = self.number_of_global_dofs()
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self):
        p = self.p
        mesh = self.mesh

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs)
        M = np.einsum('m, mij, mik, i->ijk', ws, phi, phi, self.cellmeasure, optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        # Construct the stiffness matrix
        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f, barycenter=False):
        p = self.p
        # bcs : (NQ, 3)
        # ws : (NQ, )
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        if barycenter:
            fval = f(bcs)
        else:
            pp = self.mesh.bc_to_point(bcs)
            fval = f(pp)
        phi = self.basis(bcs)
        bb = np.einsum('m, mi, mik, i->ik', ws, fval, phi, self.cellmeasure)
        cell2dof = self.dof.cell2dof
        gdof = self.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def basis(self, bc):
        """
        Compute all basis function values at a given barrycenter.
        """
        return self.mesh.space.basis(bc)

    def grad_basis(self, bc, index=np.s_[:], returncond=False):
        """
        Compute the gradients of all basis functions at a given barrycenter.
        """
        Jp, grad = self.mesh.jacobi_matrix(bc, index=index)
        Gp = np.einsum('...ijk, ...imk->...ijm', Jp, Jp)
        Gp = np.linalg.inv(Gp)
        Gp_cond = np.linalg.cond(Gp)
        grad = np.einsum('...ijk, ...imk->...imj', Gp, grad)
        grad = np.einsum('...ijk, ...imj->...imk', Jp, grad)
        if returncond:
            return grad, Gp_cond
        else:
            return grad

    def grad_basis_on_surface(self, bc, index=None):
        Js, grad, ps = self.mesh.surface_jacobi_matrix(bc, index=index)
        Gs = np.einsum('...ijk, ...imk->...ijm', Js, Js)
        Gs = np.linalg.inv(Gs)
        grad = np.einsum('...ijk, ...imk->...imj', Gs, grad)
        grad = np.einsum('...ijk, ...imj->...imk', Js, grad)
        n = np.cross(Js[..., 0, :], Js[..., 1, :], axis=-1)
        return grad, ps, n

    def hessian_basis(self, bc, index=None):
        pass

    def value(self, uh, bc, index=None):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        if index is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    def grad_value(self, uh, bc, index=np.s_[:]):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if index is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    def grad_value_on_surface(self, uh, bc, index=None):
        gphi, ps, n = self.grad_basis_on_surface(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if index is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val, ps, n

    def hessian_value(self, uh, bc, index=None):
        pass

    def div_value(self, uh, bc, index=None):
        pass

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.mesh.node

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
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def to_function(self, data):
        p = self.p
        if p == 1:
            uh = self.function(array=data)
            return uh
        elif p == 2:
            cell2dof = self.cell_to_dof()
            uh = self.function()
            uh[cell2dof] = data[:, [0, 5, 4, 1, 3, 2]]
            return uh

    def set_dirichlet_bc(self, uh, gD, threshold=None, q=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        pass

