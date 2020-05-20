import numpy as np

from ..quadrature import FEMeshIntegralAlg

class CrouzeixRaviartFiniteElementSpace():
    def __init__(self, mesh, q=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        mtype = mesh.meshtype
        if mtype == 'tri':
            self.TD = 2
        elif mtype == 'tet':
            self.TD = 3
        self.GD = mesh.geo_dimension()

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else self.p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

    def number_of_global_dofs(self):
        return self.mesh.number_of_faces()

    def number_of_local_dofs(self):
        return self.TD+1

    def interpolation_points(self):
        return self.mesh.entity_barycenter('face') 

    def cell_to_dof(self, index=None):
        cell2dof = self.mesh.ds.cell_to_face()
        index = index if index is not None else np.s_[:]
        return cell2dof[index]

    def face_to_dof(self, index=None):
        NF = self.mesh.number_of_faces()
        if index is None:
            return np.arange(NF)
        else:
            return index

    def edge_to_dof(self, index=None):
        NF = self.mesh.number_of_faces()
        if index is None:
            return np.arange(NF)
        else:
            return index

    def boundary_dof(self):
        return  self.mesh.ds.boundary_face_index()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

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
        TD = self.TD
        GD = self.GD
        mesh = self.mesh
        localFace = mesh.ds.localFace
        phi = np.prod(bc[..., localFace], axis=-1)/GD**GD
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
        TD = self.TD
        GD = self.GD
        mesh = self.mesh
        localFace = mesh.ds.localFace
        Dlambda = mesh.grad_lambda() # (NC, TD+1, GD)
        Dlambda[:, localFace, :] # (NC, TD+1, TD, GD)
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
