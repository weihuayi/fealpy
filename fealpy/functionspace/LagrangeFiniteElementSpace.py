import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from .function import Function

from .femdof import multi_index_matrix1d
from .femdof import multi_index_matrix2d
from .femdof import multi_index_matrix3d

from .femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from .femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from ..quadrature import FEMeshIntegralAlg


class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C', q=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if spacetype == 'C':
            if mesh.meshtype == 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype == 'tri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype == 'stri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype == 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.TD = 3
        elif spacetype == 'D':
            if mesh.meshtype == 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype == 'tri':
                self.dof = DPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype == 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.TD = 3

        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.multi_index_matrix = [multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=None):
        index = index if index is not None else np.s_[:]
        return self.dof.cell2dof[index]

    def face_to_dof(self, index=None):
        return self.dof.face_to_dof()

    def edge_to_dof(self, index=None):
        return self.dof.edge_to_dof()

    def boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def grad_recovery(self, uh, method='simple'):
        GD = self.GD
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        ldof = self.number_of_local_dofs()
        p = self.p
        bc = self.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = self.function(dim=GD)

        if method == 'simple':
            deg = np.bincount(cell2dof.flat, minlength = gdof)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area':
            measure = self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...',guh,measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area_harmonic':
            measure = 1/self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...',guh,measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance_harmonic':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = 1/np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)
        rguh /= deg.reshape(-1, 1)
        return rguh

    def edge_basis(self, bc, index, lidx, direction=True):
        """
        compute the basis function values at barycentric point bc on edge

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim,)` or `(NQ, tdim)`

        Returns
        -------
        phi : numpy.array
            the shape of 'phi' can be `(NE, ldof)` or `(NE, NQ, ldof)`

        See also
        --------

        Notes
        -----

        """

        mesh = self.mesh

        cell2cell = mesh.ds.cell_to_cell()
        isInEdge = (cell2cell[index, lidx] != index)

        NE = len(index)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)

        bcs[idx, ..., nmap[lidx]] = bc[..., 0]
        bcs[idx, ..., pmap[lidx]] = bc[..., 1]

        if direction == False:
            bcs[idx[isInEdge], ..., nmap[lidx[isInEdge]]] = bc[..., 1]
            bcs[idx[isInEdge], ..., pmap[lidx[isInEdge]]] = bc[..., 0]

        return self.basis(bcs)

    def edge_grad_basis(self, bc, index, lidx, direction=True):
        NE = len(index)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        p = self.p   # the degree of polynomial basis function
        TD = self.TD

        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bcs.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bcs[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_dofs()
        shape = bcs.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.mesh.grad_lambda()
        gphi = np.einsum('k...ij, kjm->k...im', R, Dlambda[index, :, :])
        return gphi

    def face_basis(self, bc):
        p = self.p   # the degree of polynomial basis function
        TD = self.TD - 1
        multiIndex = self.multi_index_matrix[TD-1](p)

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi[..., np.newaxis, :] # (..., 1, ldof)


    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, ldof)` or `(NQ, 1, ldof)`

        See Also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function

        if p == 0 and self.spacetype == 'D':
            if len(bc.shape) == 1:
                return np.ones(1, dtype=self.ftype)
            else:
                return np.ones((bc.shape[0], 1), dtype=self.ftype)

        TD = self.TD
        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi[..., np.newaxis, :] # (..., 1, ldof)

    def grad_basis(self, bc, index=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can b `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function
        TD = self.TD

        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_dofs()
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.mesh.grad_lambda()
        index = index if index is not None else np.s_[:]
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda[index, :, :])
        return gphi #(..., NC, ldof, GD)

    def value(self, uh, bc, index=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    def grad_value(self, uh, bc, index=None):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    def div_value(self, uh, bc, index=None):
        dim = len(uh.shape)
        GD = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == gdim):
            val = self.grad_value(uh, bc, index=index)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = u(ipoint)
        return self.function(dim=dim, array=uI)

    def projection(self, u):
        """
        """
        M= self.mass_matrix()
        F = self.source_vector(u)
        uh = self.function()
        uh[:] = spsolve(M, F).reshape(-1)
        return uh

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
        return np.zeros(shape, dtype=self.ftype)

    def integral_basis(self):
        """
        """
        cell2dof = self.cell_to_dof()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs)
        cc = np.einsum('m, mik, i->ik', ws, phi, self.cellmeasure)
        gdof = self.number_of_global_dofs()
        c = np.zeros(gdof, dtype=self.ftype)
        np.add.at(c, cell2dof, cc)
        return c

    def revcovery_matrix(self, rtype='simple'):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        cell = self.mesh.entity('cell')
        GD = self.GD
        cellmeasure = self.cellmeasure
        gphi = self.mesh.grad_lambda()
        G = []
        if rtype == 'simple':
            D = spdiags(1.0/np.bincount(cell.flat), 0, NN, NN)
        elif rtype == 'harmonic':
            gphi = gphi/cellmeasure.reshape(-1, 1, 1)
            d = np.zeros(NN, dtype=np.float)
            np.add.at(d, cell, 1/cellmeasure.reshape(-1, 1))
            D = spdiags(1/d, 0, NN, NN)

        I = np.einsum('k, ij->ijk', np.ones(GD+1), cell)
        J = I.swapaxes(-1, -2)
        for i in range(GD):
            val = np.einsum('k, ij->ikj', np.ones(GD+1), gphi[:, :, i])
            G.append(D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN)))
        return G

    def linear_elasticity_matrix(self, mu, lam, format='csr'):
        """
        construct the linear elasticity fem matrix
        """
        cellmeasure = self.cellmeasure
        cell2dof = self.cell_to_dof()
        GD = self.GD

        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.grad_basis(bcs)

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()

        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []
        for k, (i, j) in enumerate(idx):
            Aij = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure)
            A.append(csr_matrix((Aij.flat, (I.flat, J.flat)), shape=(gdof, gdof)))

        T = csr_matrix((gdof, gdof), dtype=self.ftype)
        D = csr_matrix((gdof, gdof), dtype=self.ftype)
        C = []
        for i in range(GD):
            D += A[imap[(i, i)]]
            C.append([T]*GD)
        D *= mu

        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    C[i][j] = D + (mu+lam)*A[imap[(i, i)]]
                else:
                    C[i][j] = lam*A[imap[(i, j)]] + mu*A[imap[(i, j)]].T
                    C[j][i] = C[i][j].T
        if format == 'csr':
            return bmat(C, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(C, format='bsr')
        elif format == 'list':
            return C

    def recovery_linear_elasticity_matrix(self, mu, lam, format='csr'):
        """
        construct the recovery linear elasticity fem matrix
        """
        G = self.revcovery_matrix()
        M = self.mass_matrix()

        cellmeasure = self.cellmeasure
        cell2dof = self.cell_to_dof()
        GD = self.GD

        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.grad_basis(bcs)

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()

        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []
        for k, (i, j) in enumerate(idx):
            A.append(G[i].T@M@G[j])

        T = csr_matrix((gdof, gdof), dtype=self.ftype)
        D = csr_matrix((gdof, gdof), dtype=self.ftype)
        C = []
        for i in range(GD):
            D += A[imap[(i, i)]]
            C.append([T]*GD)
        D *= mu
        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    C[i][j] = D + (mu+lam)*A[imap[(i, i)]]
                else:
                    C[i][j] = lam*A[imap[(i, j)]] + mu*A[imap[(i, j)]].T
                    C[j][i] = C[i][j].T
        if format == 'csr':
            return bmat(C, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(C, format='bsr')
        elif format == 'list':
            return C

    def stiff_matrix(self, cfun=None):
        p = self.p
        GD = self.geo_dimension()

        if p == 0:
            raise ValueError('The space order is 0!')

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        gphi = self.grad_basis(bcs)

        if cfun is not None:
            ps = self.mesh.bc_to_point(bcs)
            d = cfun(ps)

            if isinstance(d, (int, float)):
                dgphi = d*gphi
            elif len(d) == GD:
                dgphi = np.einsum('m, ...im->...im', d, gphi)
            elif isinstance(d, np.ndarray):
                if len(d.shape) == 1:
                    dgphi = np.einsum('i, ...imn->...imn', d, gphi)
                elif len(d.shape) == 2:
                    dgphi = np.einsum('...i, ...imn->...imn', d, gphi)
                elif len(d.shape) == 3: #TODO:
                    dgphi = np.einsum('...imn, ...in->...im', d, gphi)
                elif len(d.shape) == 4: #TODO:
                    dgphi = np.einsum('...imn, ...in->...im', d, gphi)
                else:
                    raise ValueError("The ndarray shape length should < 5!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dgphi = gphi

        # Compute the element sitffness matrix
        # ws:(NQ,)
        # dgphi: (NQ, NC, ldof, GD)
        A = np.einsum('i, ijkm, ijpm, j->jkp',
                ws, dgphi, gphi, self.cellmeasure,
                optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs()

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, cfun=None, barycenter=False):
        p = self.p
        mesh = self.mesh
        cellmeasure = self.cellmeasure

        if p == 0:
            NC = mesh.number_of_cells()
            M = spdiags(cellmeasure, 0, NC, NC)
            return M

        # bcs: (NQ, TD+1)
        # ws: (NQ, )
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        # phi: (NQ, ldof)
        phi = self.basis(bcs)

        if cfun is not None:
            if barycenter is True:
                d = cfun(bcs) # (NQ, NC)
            else:
                ps = self.mesh.bc_to_point(bcs) # (NQ, NC, GD)
                d = cfun(ps) # (NQ, NC)

            if isinstance(d, (int, float)):
                dphi = d*phi
            elif isinstance(d, np.ndarray):
                if (len(d.shape) == 1):
                    dphi = np.einsum('i, mij->mij', d, phi)
                elif len(d.shape) == 2:
                    dphi = np.einsum('mi, mij->mij', d, phi)
                else:
                    raise ValueError("The ndarray shape length should < 3!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dphi = phi
        M = np.einsum(
                'm, mij, mik, i->ijk',
                ws, dphi, phi, self.cellmeasure,
                optimize=True)

        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('ij, k->ijk',  cell2dof, np.ones(ldof))
        J = I.swapaxes(-1, -2)

        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f, dim=None):
        p = self.p
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        pp = self.mesh.bc_to_point(bcs)
        fval = f(pp)

        gdof = self.number_of_global_dofs()
        shape = gdof if dim is None else (gdof, dim)
        b = np.zeros(shape, dtype=self.ftype)

        if p > 0:
            if type(fval) in {float, int}:
                if fval == 0.0:
                    return b
                else:
                    phi = self.basis(bcs)
                    bb = np.einsum('m, mik, i->ik...', 
                            ws, phi, self.cellmeasure)
                    bb *= fval
            else:
                phi = self.basis(bcs)
                bb = np.einsum('m, mi..., mik, i->ik...',
                        ws, fval, phi, self.cellmeasure)
            cell2dof = self.cell_to_dof() #(NC, ldof)
            if dim is None:
                np.add.at(b, cell2dof, bb)
            else:
                np.add.at(b, (cell2dof, np.s_[:]), bb)
        else:
            b = np.einsum('i, ik..., k->k...', ws, fval, cellmeasure)

        return b

    def set_dirichlet_bc(self, uh, g, is_dirichlet_boundary=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        ipoints = self.interpolation_points()
        isDDof = self.boundary_dof(threshold=is_dirichlet_boundary)
        uh[isDDof] = g(ipoints[isDDof])
        return isDDof

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


