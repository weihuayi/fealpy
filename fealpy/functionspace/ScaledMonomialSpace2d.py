import numpy as np
from numpy.linalg import inv
from .function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg


class SMDof2d():
    """
    缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = self.multi_index_matrix()
        self.cell2dof = self.cell_to_dof()

    def multi_index_matrix(self, p=None):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0.

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        ldof = self.number_of_local_dofs(p=p)
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p)
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self, p=None):
        if p is None:
            p = self.p
        return (p+1)*(p+2)//2

    def number_of_global_dofs(self, p=None):
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        return NC*ldof


class ScaledMonomialSpace2d():
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh

        if bc is None:
            self.cellbarycenter = mesh.entity_barycenter('cell')
        else:
            self.cellbarycenter = bc

        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')
        self.cellsize = np.sqrt(self.cellmeasure)
        self.dof = SMDof2d(mesh, p)
        self.GD = 2

        q = q if q is not None else p+3
        self.integralalg = PolygonMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure,
                cellbarycenter=self.cellbarycenter)

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    def edge_basis(self, point, index=None, p=None):
        p = self.p if p is None else p
        center = self.integralalg.edgebarycenter
        h = self.integralalg.edgemeasure
        t = self.mesh.edge_unit_tagent()

        index = index if index is not None else np.s_[:]
        val = np.sum((point - center[index])*t[index], axis=-1)/h[index]
        phi = np.ones(val.shape + (p+1,), dtype=self.ftype)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., np.newaxis]
            np.multiply.accumulate(phi, axis=-1, out=phi)
        return phi

    def basis(self, point, index=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., M, 2), M is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., M, ldof)

        """
        if p is None:
            p = self.p
        h = self.cellsize
        NC = self.mesh.number_of_cells()

        ldof = self.number_of_local_dofs(p=p)
        if p == 0:
            shape = point.shape[:-1] + (1, )
            return np.ones(shape, dtype=np.float)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float)  # (..., M, ldof)
        index = index if index is not None else np.s_[:] 
        phi[..., 1:3] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1

        return phi

    def value(self, uh, point, index=None):
        phi = self.basis(point, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        return np.einsum(s1, phi, uh[cell2dof[index]])

    def grad_basis(self, point, index=None, p=None):
        if p is None:
            p = self.p
        h = self.cellsize
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 2)
        gphi = np.zeros(shape, dtype=np.float)
        gphi[..., 1, 0] = 1
        gphi[..., 2, 1] = 1
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            phi = self.basis(point, index=index)
            for i in range(2, p+1):
                gphi[..., start:start+i, 0] = np.einsum('i, ...i->...i', r[i-1::-1], phi[..., start-i:start])
                gphi[..., start+1:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i], phi[..., start-i:start])
                start += i+1
        index = index if index is not None else np.s_[:]
        if point.shape[-2] == len(index):
            return gphi/h[index].reshape(-1, 1, 1)
        elif point.shape[0] == len(index):
            return gphi/h[index].reshape(-1, 1, 1, 1)

    def grad_value(self, uh, point, index=None):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        index = index if index is not None else np.s_[:]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == len(index):
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return np.einsum(s1, gphi, uh[cell2dof[index]])
        elif point.shape[0] == len(index):
            return np.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]])

    def laplace_basis(self, point, index=None, p=None):
        if p is None:
            p = self.p
        area = self.cellmeasure

        ldof = self.number_of_local_dofs(p=p)

        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            phi = self.basis(point, index=index)
            for i in range(2, p+1):
                lphi[..., start:start+i-1] += np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                lphi[..., start+2:start+i+1] += np.einsum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                start += i+1
        index = index if index is not None else np.s_[:]
        return lphi/area[index].reshape(-1, 1)

    def hessian_basis(self, point, index=None, p=None):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters
        ----------
        point : numpy array
            The shape of point is (..., NC, 2)

        Returns
        -------
        hphi : numpy array
            the shape of hphi is (..., NC, ldof, 3)
        """
        if p is None:
            p = self.p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs()

        shape = point.shape[:-1]+(ldof, 3)
        hphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            phi = self.basis(point, index=index)
            for i in range(2, p+1):
                hphi[..., start:start+i-1, 0] = np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                hphi[..., start+2:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                r0 = np.arange(1, i)
                r0 = r0*r0[-1::-1]
                hphi[..., start+1:start+i, 2] = np.einsum('i, ...i->...i', r0, phi[..., start-2*i+1:start-i])
                start += i+1

        index = index if index is not None else np.s_[:]
        return hphi/area[index].reshape(-1, 1, 1)

    def laplace_value(self, uh, point, index=None):
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        index = index if index is not None else np.s_[:]
        return np.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]])

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)

    def cell_mass_matrix(self):
        return self.matrix_H()

    def edge_mass_matrix(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs)
        phi = self.edge_basis(ps, p=p)
        H = np.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, measure, optimize=True)
        return H

    def mass_matrix(self, p=None):
        return self.matrix_H(p=p)

    def stiff_matrix(self, p=None):
        gphi = self.grad_basis
        def f(x, index):
            gphi = self.grad_basis(x, index=index, p=p)
            return np.einsum('jkm, jpm->jkp', gphi, gphi)
        A = self.integralalg.integral(f, celltype=True, q=p+3)
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def matrix_H(self, p=None):
        if p is None:
            p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - self.cellbarycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - self.cellbarycenter[edge2cell[isInEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = self.number_of_local_dofs(p=p)
        H = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def projection(self, F):
        """
        F is a function in MonomialSpace2d, this function project  F to 
        ScaledMonomialSpace2d.
        """
        mspace = F.space
        C = self.matrix_C(mspace)
        H = self.matrix_H()
        PI0 = inv(H)@C
        SS = self.function()
        SS[:] = np.einsum('ikj, ij->ik', PI0, F[self.cell_to_dof()]).reshape(-1)
        return SS

    def matrix_C(self, mspace):
        def f(x, index):
            return np.einsum(
                    '...im, ...in->...imn',
                    self.basis(x, index),
                    mspace.basis(x, index)
                    )
        C = self.integralalg.integral(f, celltype=True)
        return C

    def interpolation(self, sh0, HB):
        """
         interpolation sh in space into self space.
        """
        p = self.p
        ldofs = self.number_of_local_dofs()
        mesh = self.mesh
        NC = mesh.number_of_cells()

        space0 = sh0.space
        h0 = space0.cellsize

        space1 = self
        h1 = space1.cellsize
        sh1 = space1.function()

        bc = (space1.cellbarycenter[HB[:, 0]] - space0.cellbarycenter[HB[:,
            1]])/h0[HB[:, [1]]]
        h = h1[HB[:, 0]]/h0[HB[:, 1]]

        c = sh0.reshape(-1, ldofs)
        d = sh1.reshape(-1, ldofs)

        num = np.zeros(NC, dtype=self.itype)
        np.add.at(num, HB[:, 0], 1)

        m = HB.shape[0]
        td = np.zeros((m, ldofs), dtype=self.ftype)

        td[:, 0] = c[HB[:, 1], 0] + c[HB[:, 1], 1]*bc[:, 0] + c[HB[:, 1], 2]*bc[:, 1]
        td[:, 1] = h*c[HB[:, 1], 1]
        td[:, 2] = h*c[HB[:, 1], 2]

        if p > 1:
            td[:, 0] += c[HB[:, 1], 3]*bc[:, 0]**2 + c[HB[:, 1], 4]*bc[:, 0]*bc[:, 1] + c[HB[:, 1], 5]*bc[:, 1]**2
            td[:, 1] += 2*c[HB[:, 1], 3]*bc[:, 0]*h + c[HB[:, 1], 4]*bc[:, 1]*h
            td[:, 2] += c[HB[:, 1], 4]*bc[:, 0]*h + 2*c[HB[:, 1], 5]*bc[:, 1]*h
            td[:, 3] = c[HB[:, 1], 3]*h**2
            td[:, 4] = c[HB[:, 1], 4]*h**2
            td[:, 5] = c[HB[:, 1], 5]*h**2

        np.add.at(d, (HB[:, 0], np.s_[:]), td)
        d /= num.reshape(-1, 1)
        return sh1
