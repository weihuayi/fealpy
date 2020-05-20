import numpy as np
from numpy.linalg import inv, pinv
from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class RTDof2d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TriangleMesh object
        p : the space order, p>=1

        Notes
        -----
        Here `p` begin from 1, RT_1 is the lowest degree elements (which is
        traditionally called the RT_0 element).

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 1
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*p).reshape(NE, p)
        return edge2dof

    def cell_to_dof(self):
        """
        """
        p = self.p
        mesh = self.mesh
        cell2edge = mesh.ds.cell_to_edge()

        if p == 1:
            return cell2edge
        else:
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=np.int)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            cell2dof[edge2cell[:, [0]], edge2cell[:, [2]]*p + np.arange(p)] = edge2dof
            cell2dof[edge2cell[:, [1]], edge2cell[:, [3]]*p + np.arange(p)] = edge2dof
            if p > 2:
                idof = (p-1)*p
                cell2dof[:, 3*p:] = NE*p + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return p*(p+2) 

    def number_of_global_dofs(self):
        p = self.p
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        gdof = NE*p
        if p > 1:
            gdof += NC*(p-1)*p
        return gdof 

class RaviartThomasFiniteElementSpace2d:
    def __init__(self, mesh, p, q=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        p : the space order

        Note
        ----
        RT_p : [P_{p-1}]^d(T) + [m_1, m_2]^T P_{p-1}(T)

        (p+1)*p + (p+1)*p/2 = p**2 + p + p**2/2 + p/2 = 3/2*p**2 + 3/2*p
        = 3*(p+1)*p/2

        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)

        self.dof = RTDof2d(mesh, p)

        self.integralalg = self.smspace.integralalg
        self.integrator = self.smspace.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

        self.bcoefs = self.basis_coefficients()


    def basis_coefficients(self):
        """

        Notes
        -----
        3*p + p*(p - 1) = 3*p + p**2 - p = p*(p+2) 
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        LM, RM = self.smspace.edge_cell_mass_matrix(p=p)

        A = np.zeros((NC, p*(p+2), 3*(p+1)*p//2), dtype=self.ftype)

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        n = mesh.edge_unit_normal() 

        idx = self.smspace.index1(p=p)
        x = idx['x']
        y = idx['y']

        ndof = (p+1)*p//2
        idx2 = np.arange(ndof)[None, None, :]

        idx0 = edge2cell[:, [0]][:, None, None]
        idx1 = (edge2cell[:, [2]]*p + np.arange(p))[:, :, None]
        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, 2*ndof + idx2] = n[:, 0, None, None]*LM[:, :,  x[0]] + n[:, 1, None, None]*LM[:, :, y[0]]

        idx0 = edge2cell[:, [1]][:, None, None]
        idx1 = (edge2cell[:, [3]]*p + np.arange(p))[:, :, None]

        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, 2*ndof + idx2] = n[:, 0, None, None]*RM[:, :,  x[0]] + n[:, 1, None, None]*RM[:, :, y[0]]

        if p == 1:
            return inv(A)
        else:
            M = self.smspace.mass_matrix(p=p-1)
            idx = self.smspace.index1(p=p-1)
            x = idx['x']
            y = idx['y']
            idof = p*(p-1)//2
            start = 3*p
            idx1 = np.arange(3*p, 3*p+idof)[:, None]
            A[:, idx1, 0*ndof + np.arange(ndof)] = M[:, :idof, :]
            A[:, idx1, 2*ndof + np.arange(ndof)] = M[:,  x[0], :]

            idx1 = np.arange(3*p+idof, 3*p+2*idof)[:, None]
            A[:, idx1, 1*ndof + np.arange(ndof)] = M[:, :idof, :]
            A[:, idx1, 2*ndof + np.arange(ndof)] = M[:,  y[0], :]
            
            B = A.swapaxes(-1, -2)
            B = B@inv(A@B)
            return B
        
    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(3,)` or `(NQ, 3)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(NC, ldof, 2)` or `(NQ, NC, ldof, 2)`

        See Also
        --------

        Notes
        -----

        ldof = p*(p+2)
        ndof = (p+1)*p/2

        """
        p = self.p
        ndof = (p+1)*p//2
        ldof = self.number_of_local_dofs()

        idx = self.smspace.index1(p=p)
        x = idx['x']
        y = idx['y']

        c = self.bcoefs # (NC, 3*ndof, ldof) 
        ps = self.mesh.bc_to_point(bc)
        shape = ps.shape[:-1] + (ldof, 2)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)
        val = self.smspace.basis(ps) # (NQ, NC, ndof)
        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 0*ndof:1*ndof, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 1*ndof:2*ndof, :])
        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[...,  x[0]], c[:, 2:ndof:3*ndof, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[...,  y[0]], c[:, 2:ndof:3*ndof, :])
        return phi


    def grad_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        shape = (NC, ldof, 2, 2)
        gradPhi = np.zeros(shape, dtype=np.float)

        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)
        Rlambda= mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            A = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 1, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 2, :, :] = A - B

            gradPhi *= cell2edgeSign.reshape(-1, 3, 1, 1)
        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        divPhi = np.zeros((NC, ldof), dtype=np.float)
        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)

        Rlambda = mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :], axis=1) - np.sum(Dlambda[:, 2, :]*Rlambda[:, 1, :], axis=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :], axis=1) - np.sum(Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :], axis=1) - np.sum(Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi *= cell2edgeSign
        else:
            #TODO:raise a error
            print("error")

        return divPhi

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
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
        s1 = '...ijmn, ij{}->...i{}mn'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        val = self.grad_value(uh, bc, cellidx=None)
        return val.trace(axis1=-2, axis2=-1)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def interpolation(self, u, returnfun=False):
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        NE = mesh.number_of_edges()
        n = mesh.edge_unit_normal()
        l = mesh.entity_measure('edge')

        qf = IntervalQuadrature(3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        points = np.einsum('kj, ijm->kim', bcs, node[edge])
        val = u(points)
        uh = np.einsum('k, kim, im, i->i', ws, val, n, l)

        if returnfun is True:
            return Function(self, array=uh)
        else:
            return uh

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)
