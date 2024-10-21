import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .Function import Function
from ..quadrature.FEMeshIntegralAlg import FEMeshIntegralAlg

class QuadBilinearFiniteElementSpace():

    def __init__(self, mesh, q=None):

        self.mesh = mesh
        # make the vertex with smallest angle as the first vertex
        angle = self.mesh.angle()
        idx = np.argmin(angle, axis=-1)
        self.mesh.reorder_cell(idx)

        self.cellmeasure = self.mesh.entity_measure('cell')


        self.ccenter  = self.mesh.entity_barycenter('cell')
        self.bvector = self.cell_basis_vector()
        self.pvector, self.cmatrix = self.cell_parameters()
        self.bcoefs = self.basis_coefficients()

        self.GD = 2
        self.TD = 2

        q = q if q is not None else 4
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def number_of_global_dofs(self):
        return self.mesh.number_of_nodes()

    def number_of_local_dofs(self):
        return 4

    def interpolation_points(self):
        return self.mesh.entity('node')

    def cell_to_dof(self):
        return self.mesh.entity('cell')

    def boundary_dof(self):
        return self.mesh.ds.boundary_node_flag()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def cell_basis_vector(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')
        edgeCenter = mesh.entity_barycenter('edge')
        cell2edge = mesh.ds.cell_to_edge()

        bv = np.zeros((NC, GD, GD), dtype=mesh.ftype)
        bv[:, 0, :] = edgeCenter[cell2edge[:, 3]] - self.ccenter
        bv[:, 1, :] = edgeCenter[cell2edge[:, 0]] - self.ccenter
        return bv

    def cell_parameters(self):
        """
        Compute the parameters (alpha, beta), and the transform matrix from xy
        coordinate to rOs coordinate.
        """
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        A = np.zeros((NC, GD, GD), dtype=mesh.ftype)

        A[:, [0, 1], [0, 1]]  = np.sum(self.bvector**2, axis=-1)
        A[:, 0, 1] = np.sum(self.bvector[:, 0, :]*self.bvector[:, 1, :],
                axis=-1)
        A[:, 1, 0] = A[:, 0, 1]
        cmatrix = np.linalg.inv(A)

        b = np.einsum('ijk, ik->ij', self.bvector, node[cell[:, 0]] - self.ccenter)
        return np.einsum('ijk, ik->ij', cmatrix, b) - 1, cmatrix

    def basis_coefficients(self):
        """
        Compute the coefficients
        """
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        A = np.zeros((NC, 4, 4), dtype=mesh.ftype)

        pv = self.pvector # NCx2
        A[:, 0, 0] = pv[:, 0] + pv[:, 1] - 1
        A[:, 0, 1] = (pv[:, 1] - 1)*(-pv[:, 0] + pv[:, 1] + 1)
        A[:, 0, 2] = (pv[:, 0] - 1)*(pv[:, 0] - pv[:, 1] + 1)
        A[:, 0, 3] = -(pv[:, 0] - 1)*(pv[:, 1] - 1)*(pv[:, 0] + pv[:, 1] + 1)

        A[:, 1, 0] = -pv[:, 0] + pv[:, 1] + 1
        A[:, 1, 1] = -(pv[:, 1] + 1)*(pv[:, 0] + pv[:, 1] - 1)
        A[:, 1, 2] = (pv[:, 0] - 1)*(pv[:, 0] + pv[:, 1] + 1)
        A[:, 1, 3] = (pv[:, 0] - 1)*(pv[:, 1] + 1)*(pv[:, 0] - pv[:, 1] + 1)

        A[:, 2, 0] = -pv[:, 0] - pv[:, 1] - 1
        A[:, 2, 1] = (pv[:, 1] + 1)*(pv[:, 0] - pv[:, 1] + 1)
        A[:, 2, 2] = (pv[:, 0] + 1)*(-pv[:, 0] + pv[:, 1] + 1)
        A[:, 2, 3] = (pv[:, 0] + 1)*(pv[:, 1] + 1)*(pv[:, 0] + pv[:, 1] - 1)

        A[:, 3, 0] = pv[:, 0] - pv[:, 1] + 1
        A[:, 3, 1] = (pv[:, 1] - 1)*(pv[:, 0] + pv[:, 1] + 1)
        A[:, 3, 2] = -(pv[:, 0] + 1)*(pv[:, 0] + pv[:, 1] - 1)
        A[:, 3, 3] = (pv[:, 0] + 1)*(pv[:, 1] - 1)*(-pv[:, 0] + pv[:, 1] + 1)
        d = 4*(pv[:, 0]**2 + pv[:, 1]**2 - 1)

        return A/d.reshape(-1, 1, 1)


    def basis(self, bc, cellidx=None):
        """
        Compute the basis

        bc: NQx4
        bvector: NC x 2 x 2
        cmatrix: NC x 2 x 2
        """

        mesh = self.mesh
        ldof = self.number_of_local_dofs()
        v = mesh.bc_to_point(bc) - self.ccenter # NQ x NC x 2
        v = np.einsum('imn, ...in->...im', self.bvector, v) # NQ x NC x 2
        xe = np.einsum('imn, ...in->...im', self.cmatrix, v) # NQ x NC x 2
        shape = v.shape[:-1] + (ldof, )
        phi = np.zeros(shape, dtype=mesh.ftype) # 

        phi[..., 0] = (
                self.bcoefs[:, 0, 0]*xe[..., 0]*xe[..., 1] +
                self.bcoefs[:, 0, 1]*xe[..., 0] +
                self.bcoefs[:, 0, 2]*xe[..., 1] +
                self.bcoefs[:, 0, 3]
                )
        phi[..., 1] = (
                self.bcoefs[:, 1, 0]*xe[..., 0]*xe[..., 1] +
                self.bcoefs[:, 1, 1]*xe[..., 0] +
                self.bcoefs[:, 1, 2]*xe[..., 1] +
                self.bcoefs[:, 1, 3]
                )
        phi[..., 2] = (
                self.bcoefs[:, 2, 0]*xe[..., 0]*xe[..., 1] +
                self.bcoefs[:, 2, 1]*xe[..., 0] +
                self.bcoefs[:, 2, 2]*xe[..., 1] +
                self.bcoefs[:, 2, 3]
                )
        phi[..., 3] = (
                self.bcoefs[:, 3, 0]*xe[..., 0]*xe[..., 1] +
                self.bcoefs[:, 3, 1]*xe[..., 0] +
                self.bcoefs[:, 3, 2]*xe[..., 1] +
                self.bcoefs[:, 3, 3]
                )
        return phi

    def grad_basis(self, bc, cellidx=None):
        """
        bc: NQx4
        bvector: NC x 2 x 2
        cmatrix: NC x 2 x 2
        """

        mesh = self.mesh
        ldof = self.number_of_local_dofs()
        GD = self.geo_dimension()
        v = mesh.bc_to_point(bc) - self.ccenter # NQ x NC x 2
        v = np.einsum('imn, ...in->...im', self.bvector, v) # NQ x NC x 2
        xe = np.einsum('imn, ...in->...im', self.cmatrix, v) # NQ x NC x 2

        ctheta = np.cross(self.bvector[:, 0, :], self.bvector[:, 1, :])
        grad = np.zeros(self.bvector.shape, dtype=np.float)
        grad[:, 0, 0] = self.bvector[:, 1, 1]
        grad[:, 0, 1] = -self.bvector[:, 1, 0]
        grad[:, 1, 0] = -self.bvector[:, 0, 1]
        grad[:, 1, 1] = self.bvector[:, 0, 0]

        grad[:, 0, :] /= ctheta.reshape(-1, 1)
        grad[:, 1, :] /= ctheta.reshape(-1, 1)
        shape = v.shape[:-1] + (ldof, GD)
        gphi = np.zeros(shape, dtype=mesh.ftype)
        gxe = (
                xe[..., 0, np.newaxis]*grad[np.newaxis, :, 1, :] +
                xe[..., 1, np.newaxis]*grad[np.newaxis, :, 0, :]
                )
        gphi[..., 0, :] = (
                self.bcoefs[np.newaxis, :, 0, 0, np.newaxis]*gxe +
                self.bcoefs[np.newaxis, :, 0, 1, np.newaxis]*grad[np.newaxis, :, 0, :] +
                self.bcoefs[np.newaxis, :, 0, 2, np.newaxis]*grad[np.newaxis, :, 1, :]
                )
        gphi[..., 1, :] = (
                self.bcoefs[np.newaxis, :, 1, 0, np.newaxis]*gxe +
                self.bcoefs[np.newaxis, :, 1, 1, np.newaxis]*grad[np.newaxis, :, 0, :] +
                self.bcoefs[np.newaxis, :, 1, 2, np.newaxis]*grad[np.newaxis, :, 1, :]
                )
        gphi[..., 2, :] = (
                self.bcoefs[np.newaxis, :, 2, 0, np.newaxis]*gxe +
                self.bcoefs[np.newaxis, :, 2, 1, np.newaxis]*grad[np.newaxis, :, 0, :] +
                self.bcoefs[np.newaxis, :, 2, 2, np.newaxis]*grad[np.newaxis, :, 1, :]
                )
        gphi[..., 3, :] = (
                self.bcoefs[np.newaxis, :, 3, 0, np.newaxis]*gxe +
                self.bcoefs[np.newaxis, :, 3, 1, np.newaxis]*grad[np.newaxis, :, 0, :] +
                self.bcoefs[np.newaxis, :, 3, 2, np.newaxis]*grad[np.newaxis, :, 1, :]
                )
        return gphi


    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
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

    def interpolation(self, u, dim=None):
        ipoint = self.interpolation_points()
        uI = Function(self, dim=dim)
        uI[:] = u(ipoint)
        return uI

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def stiff_matrix(self, cfun=None):
        GD = self.mesh.geo_dimension()

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        gphi = self.grad_basis(bcs)

        # Compute the element sitffness matrix
        A = np.einsum('i, ijkm, ijpm, j->jkp', ws, gphi, gphi, self.cellmeasure, optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs()

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def source_vector(self, f):
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        pp = self.mesh.bc_to_point(bcs)
        fval = f(pp)

        phi = self.basis(bcs)
        bb = np.einsum('i, ik, ik..., k->k...', ws, fval, phi, self.cellmeasure)
        cell2dof = self.cell_to_dof()
        print(cell2dof.shape)
        print(bb.shape)
        gdof = self.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b
