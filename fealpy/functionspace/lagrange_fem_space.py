import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .function import Function
from .dof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from .dof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d


class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p
        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]
        if spacetype is 'C':
            if mesh.meshtype is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'stri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.TD = 3
        elif spacetype is 'D':
            if mesh.meshtype is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = DPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.TD = 3

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def edge_basis(self, bc, cellidx, lidx, direction=True):
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
        isInEdge = (cell2cell[cellidx, lidx] != cellidx)

        NE = len(cellidx)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)

        bcs[idx, ..., nmap[lidx]] = bc[..., 0]
        bcs[idx, ..., pmap[lidx]] = bc[..., 1]

        if direction is False:
            bcs[idx[isInEdge], ..., nmap[lidx[isInEdge]]] = bc[..., 1]
            bcs[idx[isInEdge], ..., pmap[lidx[isInEdge]]] = bc[..., 0]

        return self.basis(bcs)

    def edge_grad_basis(self, bc, cellidx, lidx, direction=True):
        NE = len(cellidx)
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
        gphi = np.einsum('k...ij, kjm->k...im', R, Dlambda[cellidx, :, :])
        return gphi

    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`
        Returns
        -------
        phi : numpy.array
            the shape of 'phi' can be `(ldof, )` or `(NQ, ldof)`

        See also
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
        return phi

    def grad_basis(self, bc, cellidx=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`

        Returns
        -------
        gphi : numpy.array
            the shape of `gphi` can b `(NC, ldof, gdim)' or
            `(NQ, NC, ldof, gdim)'

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
        if cellidx is None:
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
        else:
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda[cellidx, :, :])
        return gphi

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...j, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        dim = len(uh.shape)
        gdim = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == gdim):
            val = self.grad_value(uh, bc, cellidx=cellidx)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = Function(self, dim=dim)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

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

    def stiff_matrix(self, qf, cellmeasure, cfun=None):
        p = self.p
        GD = self.mesh.geo_dimension()

        if p == 0:
            raise ValueError('The space order is 0!')

        bcs, ws = qf.get_quadrature_points_and_weights()
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
        A = np.einsum('i, ijkm, ijpm, j->jkp', ws, dgphi, gphi, cellmeasure, optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs()

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, qf, cellmeasure, cfun=None, barycenter=False):
        p = self.p
        mesh = self.mesh
        if p == 0:
            NC = mesh.number_of_cells()
            M = spdiags(cellmeasure, 0, NC, NC)
            return M

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.basis(bcs)

        if cfun is not None:
            if barycenter is True:
                d = cfun(bcs)
            else:
                ps = self.mesh.bc_to_point(bcs)
                d = cfun(ps)

            if isinstance(d, (int, float)):
                dphi = d*phi
            elif isinstance(d, np.ndarray):
                if (len(d.shape) == 1):
                    dphi = np.einsum('i, mj->mij', d, phi)
                elif len(d.shape) == 2:
                    dphi = np.einsum('mi, mj->mij', d, phi)
                else:
                    raise ValueError("The ndarray shape length should < 3!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dphi = phi

        if len(dphi.shape) == 2:
            M = np.einsum(
                    'm, mj, mk, i->ijk',
                    ws, dphi, phi, cellmeasure,
                    optimize=True)
        elif len(dphi.shape) == 3:
            M = np.einsum(
                    'm, mij, mk, i->ijk',
                    ws, dphi, phi, cellmeasure,
                    optimize=True)

        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f, qf, measure, surface=None):
        p = self.p

        bcs, ws = qf.quadpts, qf.weights
        pp = self.mesh.bc_to_point(bcs)

        if surface is not None:
            pp, _ = surface.project(pp)
        fval = f(pp)

        if p > 0:
            phi = self.basis(bcs)
            bb = np.einsum('i, ik, i..., k->k...', ws, fval, phi, measure)
            cell2dof = self.dof.cell2dof
            gdof = self.number_of_global_dofs()
            b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        else:
            b = np.einsum('i, ik, k->k', ws, fval,  measure)
        return b


class VectorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(
                mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.TD = self.scalarspace.TD
        self.GD = self.scalarspace.GD

    def __str__(self):
        return "Vector Lagrange finite element space!"

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def vector_dim(self):
        return self.GD

    def cell_to_dof(self):
        GD = self.GD
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = GD*cell2dof + np.arange(GD)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof_flag(self):
        GD = self.GD
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, GD)

    def number_of_global_dofs(self):
        return self.GD*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.GD*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        GD = self.GD
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(self.GD))
        shape += [-1, GD] 
        phi = phi.reshape(shape)
        return phi

    def div_basis(self, bcs, cellidx=None):
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1]
        return gphi.reshape(shape)

    def value(self, uh, bcs, cellidx=None):
        phi = self.basis(bcs)
        cell2dof = self.cell_to_dof()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...jm, ij->...im',  phi, uh) 
        return val 

    def div_value(self, uh, bcs, cellidx=None):
        dphi = self.div_basis(bcs, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...j, ij->...i',  dphi, uh) 
        return val

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=self.mesh.ftype)

    def interpolation(self, u):
        GD = self.GD
        c2d = self.dof.cell2dof
        ldof = self.dof.number_of_local_dofs()
        cell2dof = self.cell_to_dof().reshape(-1, ldof, GD)
        p = self.dof.interpolation_points()[c2d]
        uI = Function(self)
        uI[cell2dof] = u(p)
        return uI

    def stiff_matrix(self, qf, measure):
        p = self.p
        mesh = self.mesh
        GD = self.GD
        S = self.scalarspace.stiff_matrix(qf, measure)

        I, J = np.nonzero(S)
        gdof = self.number_of_global_dofs()
        A = coo_matrix(gdof, gdof)
        for i in range(self.GD):
            A += coo_matrix((S.data, (GD*I + i, GD*J + i)), shape=(gdof, gdof), dtype=mesh.ftype)
        return A.tocsr() 

    def mass_matrix(self, qf, measure, cfun=None, barycenter=True):
        p = self.p
        mesh = self.mesh
        GD = self.GD

        M = self.scalarspace.mass_matrix(qf, measure, cfun=cfun, barycenter=barycenter)
        I, J = np.nonzero(M)
        gdof = self.number_of_global_dofs()
        A = coo_matrix(gdof, gdof)
        for i in range(self.GD):
            A += coo_matrix((M.data, (GD*I + i, GD*J + i)), shape=(gdof, gdof), dtype=mesh.ftype)
        return A.tocsr() 

    def source_vector(self, f, qf, measure, surface=None):
        p = self.p
        mesh = self.mesh
        GD = self.GD       
        
        bcs, ws = qf.quadpts, qf.weights
        pp = self.mesh.bc_to_point(bcs)
        
        if surface is not None:
            pp, _ = surface.project(pp)

        fval = f(pp)
        if p > 0:
            phi = self.scalarspace.basis(bcs)
            cell2dof = self.dof.cell2dof
            gdof = self.dof.number_of_global_dofs()
            b = np.zeros((gdof, GD), dtype=mesh.ftype)
            for i in range(GD):
                bb = np.einsum('i, ik, i..., k->k...', ws, fval[..., i], phi, measure)
                b[:, i]  = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        else:
            b = np.einsum('i, ikm, k->km', ws, fval,  measure)

        return b.reshape(-1)


class SymmetricTensorLagrangeFiniteElementSpace():
    #TODO: improve it 
    def __init__(self, mesh, p, spacetype='C'):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype=spacetype)
        self.mesh = mesh
        self.p = p
        self.dof = self.scalarspace.dof
        self.GD = self.scalarspace.GD
        self.TD = self.scalarspace.TD

        if self.TD == 2:
            self.T = np.array([[(1, 0), (0, 0)], [(0, 1), (1, 0)], [(0, 0), (0, 1)]])
        elif self.dim == 3:
            self.T = np.array([
                [(1, 0, 0), (0, 0, 0), (0, 0, 0)], 
                [(0, 1, 0), (1, 0, 0), (0, 0, 0)],
                [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
                [(0, 0, 0), (0, 1, 0), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
                [(0, 0, 0), (0, 0, 0), (0, 0, 1)]])

    def __str__(self):
        return " Symmetric Tensor Lagrange finite element space!"

    def geom_dim(self):
        return self.dim

    def tensor_dim(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def cell_to_dof(self):
        tdim = self.tensor_dim()
        cell2dof = self.dof.cell2dof[..., np.newaxis]
        cell2dof = tdim*cell2dof + np.arange(tdim)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def boundary_dof(self):
        tdim = self.tensor_dim()
        isBdDof = self.dof.boundary_dof()
        return np.repeat(isBdDof, tdim)

    def number_of_global_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        tdim = self.tensor_dim()
        return tdim*self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bcs):
        dim = self.dim
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mno->...jmno', phi0, self.T)
        shape += [-1, dim, dim]
        return phi.reshape(shape)

    def div_basis(self, bcs, cellidx=None):
        dim = self.dim
        gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1, dim]
        return gphi.reshape(shape)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        uh0 = uh.reshape(-1, self.dim)
        if cellidx is None:
            uh0 = uh0[cell2dof].reshape(-1)
        else:
            uh0 = uh0[cell2dof[cellidx]].reshape(-1)
        val = np.einsum('...jm, ij->...im',  phi, uh0) 
        return val 

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)

    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        uI = Function(self)
        uI[:] = u(ipoint).flat[:]
        return uI
