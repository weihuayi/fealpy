import numpy as np
from numpy.linalg import inv, pinv
from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d
from .femdof import multi_index_matrix2d

class RTDof2d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TriangleMesh object
        p : the space order, p>=0

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 0
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
        edof = self.p + 1
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*edof).reshape(NE, edof)
        return edge2dof

    def cell_to_dof(self):
        """
        """
        p = self.p 
        mesh = self.mesh
        cell2edge = mesh.ds.cell_to_edge()

        if p == 0:
            return cell2edge
        else:
            edof = p + 1
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=np.int)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            cell2dof[edge2cell[:, [0]], edge2cell[:, [2]]*edof + np.arange(edof)] = edge2dof
            cell2dof[edge2cell[:, [1]], edge2cell[:, [3]]*edof + np.arange(edof)] = edge2dof
            if p > 1:
                idof = (p+1)*p
                cell2dof[:, 3*edof:] = NE*edof+ np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return (p+1)*(p+3) 

    def number_of_global_dofs(self):
        p = self.p
        
        edof = p + 1
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        gdof = NE*edof
        if p > 0:
            gdof += NC*(p+1)*p
        return gdof 

class RaviartThomasFiniteElementSpace2d:
    def __init__(self, mesh, p, q=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        p : the space order, p>=0

        Note
        ----
        RT_p : [P_{p-1}]^d(T) + [m_1, m_2]^T \\bar P_{p-1}(T)

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

    def cell_to_edge_sign(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge2cell
        NC = mesh.number_of_cells()
        cell2edgeSign = -np.ones((NC, 3), dtype=np.int)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = 1
        return cell2edgeSign

    def basis_coefficients(self):
        """

        Notes
        -----
        3*(p+1) + 2*(p+1)*p/2 = (p+1)*(p+3) 
        """
        p = self.p
        ldof = self.number_of_local_dofs()
        ndof = self.smspace.number_of_local_dofs()
        edof = p + 1

        mesh = self.mesh
        NC = mesh.number_of_cells()

        LM, RM = self.smspace.edge_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        n = mesh.edge_unit_normal() 

        idx2 = np.arange(ndof)[None, None, :]
        idx3 = np.arange(2*ndof, 2*ndof+edof)[None, None, :]

        idx0 = edge2cell[:, [0]][:, None, None]
        idx1 = (edge2cell[:, [2]]*edof + np.arange(edof))[:, :, None]

        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  ndof:ndof+edof] + n[:, 1, None, None]*LM[:, :, ndof+1:]

        idx0 = edge2cell[:, [1]][:, None, None]
        idx1 = (edge2cell[:, [3]]*edof + np.arange(edof))[:, :, None]

        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  ndof:ndof+edof] + n[:, 1, None, None]*RM[:, :, ndof+1:]

        if p > 0:
            M = self.smspace.mass_matrix()
            idx = self.smspace.index1()
            x = idx['x']
            y = idx['y']
            idof = (p+1)*p//2
            idx1 = np.arange(3*edof, 3*edof+idof)[:, None]
            A[:, idx1, 0*ndof + np.arange(ndof)] = M[:, :idof, :]
            A[:, idx1, 2*ndof:] = M[:,  x[0], ndof-edof:]

            idx1 = np.arange(3*edof+idof, 3*edof+2*idof)[:, None]
            A[:, idx1, 1*ndof + np.arange(ndof)] = M[:, :idof, :]
            A[:, idx1, 2*ndof:] = M[:,  y[0], ndof-edof:]

        print(n)
        print(edge)
        print(A)
        
        return inv(A)

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
        ldof = self.number_of_local_dofs()

        mesh = self.mesh

        if False:
            NC = mesh.number_of_cells()
            Rlambda = mesh.rot_lambda()
            cell2edgeSign = self.cell_to_edge_sign()
            shape = bc.shape[:-1] + (NC, ldof, 2)
            phi = np.zeros(shape, dtype=np.float)
            phi[..., 0, :] = bc[..., 1, None, None]*Rlambda[:, 2, :] - bc[..., 2, None, None]*Rlambda[:, 1, :]
            phi[..., 1, :] = bc[..., 2, None, None]*Rlambda[:, 0, :] - bc[..., 0, None, None]*Rlambda[:, 2, :]
            phi[..., 2, :] = bc[..., 0, None, None]*Rlambda[:, 1, :] - bc[..., 1, None, None]*Rlambda[:, 0, :]
            phi *= cell2edgeSign.reshape(-1, 3, 1)
        else:
            ps = mesh.bc_to_point(bc)
            val = self.smspace.basis(ps, p=p+1) # (NQ, NC, ndof)
            edof = p + 1
            ndof = (p+2)*(p+1)//2

            shape = ps.shape[:-1] + (ldof, 2)
            phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)

            c = self.bcoefs # (NC, ldof, ldof) 
            x = np.arange(ndof, ndof+edof)
            y = x + 1
            phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 0*ndof:1*ndof, :])
            phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 1*ndof:2*ndof, :])
            phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., x], c[:, 2:ndof:, :])
            phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., y], c[:, 2:ndof:, :])
        return phi


    def grad_basis(self, bc):
        p = self.p
        ldof = self.number_of_local_dofs()

        mesh = self.mesh
        if p == 1:
            NC = mesh.number_of_cells()
            shape = (NC, ldof, 2, 2)
            gradPhi = np.zeros(shape, dtype=np.float)

            cell2edgeSign = self.cell_to_edge_sign()
            W = np.array([[0, 1], [-1, 0]], dtype=np.float)
            Rlambda= mesh.rot_lambda()
            Dlambda = mesh.grad_lambda()

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
            if len(bc.shape) > 1:
                return gradPhi[None, ...]
            else:
                return gradPhi
        else:
            print("error")
            return None


    def div_basis(self, bc):
        p = self.p
        ldof = self.number_of_local_dofs()

        if p == 1:
            mesh = self.mesh
            NC = mesh.number_of_cells()
            divPhi = np.zeros((NC, ldof), dtype=np.float)
            cell2edgeSign = self.cell_to_edge_sign()
            W = np.array([[0, 1], [-1, 0]], dtype=np.float)

            Rlambda = mesh.rot_lambda()
            Dlambda = mesh.grad_lambda()
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :], axis=1) - np.sum(Dlambda[:, 2, :]*Rlambda[:, 1, :], axis=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :], axis=1) - np.sum(Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :], axis=1) - np.sum(Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi *= cell2edgeSign
            if len(bc.shape) > 1:
                return divPhi[None, ...]
            else:
                return divPhi
        else:
            print("error")
            return None 


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

    def show_basis(self, fig, index=0, box=None):
        """
        Plot quvier graph for every basis in a fig object
        """

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix2d(4)/4
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)

        for i in range(ldof):
            axes = fig.add_subplot(1, ldof, i+1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], 
                    units='xy')

