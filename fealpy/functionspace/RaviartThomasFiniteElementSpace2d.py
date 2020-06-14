import numpy as np
from numpy.linalg import inv
from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

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
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        edge2dof = np.arange(NE*edof).reshape(NE, edof)
        return edge2dof

    def cell_to_dof(self):
        """
        """
        p = self.p 
        mesh = self.mesh
        if p == 0:
            cell2edge = mesh.ds.cell_to_edge()
            return cell2edge
        else:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs('all')  # 单元上的所有自由度
            cdof = self.number_of_local_dofs('cell') # 单元内部的自由度
            edof = self.number_of_local_dofs('edge') # 边内部的自由度
            cell2dof = np.zeros((NC, ldof), dtype=np.int_)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            cell2dof[edge2cell[:, [0]], edge2cell[:, [2]]*edof + np.arange(edof)] = edge2dof
            cell2dof[edge2cell[:, [1]], edge2cell[:, [3]]*edof + np.arange(edof)] = edge2dof
            cell2dof[:, 3*edof:] = NE*edof+ np.arange(NC*cdof).reshape(NC, cdof)
            return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+3) 
        elif doftype == 'cell': # number of dofs inside the cell 
            return p*(p+1) 
        elif doftype in {'face', 'edge'}: # number of dofs on a edge 
            return p+1
        elif doftype == 'node': # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        p = self.p

        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge') 
        gdof = NE*edof
        if p > 0:
            NC = self.mesh.number_of_cells()
            cdof = self.number_of_local_dofs('cell')
            gdof += NC*cdof
        return gdof 

class RaviartThomasFiniteElementSpace2d:
    def __init__(self, mesh, p, q=None, dof=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        p : the space order, p>=0
        q : the index of quadrature fromula
        dof : the object for degree of freedom

        Note
        ----
        RT_p : [P_{p}]^d(T) + [m_1, m_2]^T \\bar P_{p}(T)

        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)

        if dof is None:
            self.dof = RTDof2d(mesh, p)
        else:
            self.dof = dof

        self.integralalg = self.smspace.integralalg
        self.integrator = self.smspace.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

        self.bcoefs = self.basis_coefficients()

    def basis_coefficients(self):
        """

        Notes
        -----
        3*(p+1) + 2*(p+1)*p/2 = (p+1)*(p+3) 
        """
        p = self.p
        ldof = self.number_of_local_dofs('all')  # 单元上全部自由度的个数
        edof = self.number_of_local_dofs('edge') # 每条边内部自由度个数
        cdof = self.number_of_local_dofs('cell') # 每个单元内部自由度个数
        ndof = self.smspace.number_of_local_dofs(p=p) 

        mesh = self.mesh
        NC = mesh.number_of_cells()
        
        LM, RM = self.smspace.edge_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        n = mesh.edge_unit_normal() 

        idx2 = np.arange(ndof)[None, None, :]
        idx3 = np.arange(2*ndof, 2*ndof+edof)[None, None, :]

        # idx0 = edge2cell[:, [0]][:, None, None] this is a bug!!!
        idx0 = edge2cell[:, 0][:, None, None]
        idx1 = (edge2cell[:, [2]]*edof + np.arange(edof))[:, :, None]

        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*LM[:, :, :ndof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  ndof:ndof+edof] + n[:, 1, None, None]*LM[:, :, ndof+1:]

        # idx0 = edge2cell[:, [1]][:, None, None] this is a bug!!!
        idx0 = edge2cell[:, 1][:, None, None]
        idx1 = (edge2cell[:, [3]]*edof + np.arange(edof))[:, :, None]

        A[idx0, idx1, 0*ndof + idx2] = n[:, 0, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, 1*ndof + idx2] = n[:, 1, None, None]*RM[:, :, :ndof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  ndof:ndof+edof] + n[:, 1, None, None]*RM[:, :, ndof+1:]

        if p > 0:
            M = self.smspace.mass_matrix()
            idx = self.smspace.diff_index_1()
            x = idx['x']
            y = idx['y']
            idof = (p+1)*p//2
            idx1 = np.arange(3*edof, 3*edof+idof)[:, None]
            A[:, idx1, 0*ndof + np.arange(ndof)] = M[:, :idof, :]
            print("A:", A[:, idx1, 2*ndof + np.arange(edof)].shape)
            print("M:",  M[:,  x[0], ndof-edof:].shape)
            A[:, idx1, 2*ndof + np.arange(edof)] = M[:,  x[0], ndof-edof:]

            idx1 = np.arange(3*edof+idof, 3*edof+2*idof)[:, None]
            A[:, idx1, 1*ndof + np.arange(ndof)] = M[:, :idof, :]
            A[:, idx1, 2*ndof + np.arange(edof)] = M[:,  y[0], ndof-edof:]

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
        ldof = self.number_of_local_dofs('all')
        edof = self.number_of_local_dofs('edge') 
        ndof = self.smspace.number_of_local_dofs(p=p) 

        mesh = self.mesh
        ps = mesh.bc_to_point(bc)
        val = self.smspace.basis(ps, p=p+1) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, 2)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)

        c = self.bcoefs # (NC, ldof, ldof) 
        x = np.arange(ndof, ndof+edof)
        y = x + 1
        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 0*ndof:1*ndof, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., :ndof], c[:, 1*ndof:2*ndof, :])
        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., x], c[:, 2*ndof:, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., y], c[:, 2*ndof:, :])
        return phi

    def div_basis(self, bc):
        p = self.p
        ldof = self.number_of_local_dofs('all')
        edof = self.number_of_local_dofs('edge') 
        ndof = self.smspace.number_of_local_dofs(p=p) 

        mesh = self.mesh
        ps = mesh.bc_to_point(bc)
        val = self.smspace.grad_basis(ps, p=p+1) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, )
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof)
        c = self.bcoefs # (NC, ldof, ldof) 
        x = np.arange(ndof, ndof+edof)
        y = x + 1
        phi[:] += np.einsum('ijm, jmn->ijn', val[..., :ndof, 0], c[:, 0*ndof:1*ndof, :])
        phi[:] += np.einsum('ijm, jmn->ijn', val[..., :ndof, 1], c[:, 1*ndof:2*ndof, :])
        phi[:] += np.einsum('ijm, jmn->ijn', val[..., x, 0], c[:, 2*ndof:, :])
        phi[:] += np.einsum('ijm, jmn->ijn', val[..., y, 1], c[:, 2*ndof:, :])
        return phi

    def grad_basis(self, bc):
        pass

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    def value(self, uh, bc, index=None):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof])
        return val

    def div_value(self, uh, bc, index=None):
        dphi = self.div_basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, dphi, uh[cell2dof])
        return val

    def grad_value(self, uh, bc, index=None):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def interpolation(self, u):
        p = self.p
        mesh = self.mesh

        uh = self.function()
        edge2dof = self.dof.edge_to_dof() 
        en = mesh.edge_unit_normal()
        def f0(bc):
            ps = mesh.bc_to_point(bc, etype='edge')
            return np.einsum('ijk, jk, ijm->ijm', u(ps), en, self.smspace.edge_basis(ps))
        uh[edge2dof] = self.integralalg.edge_integral(f0, edgetype=True)

        if p >= 1:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            edof = self.number_of_local_dofs('edge')
            idof = self.number_of_local_dofs('cell') # dofs inside the cell 
            cell2dof = NE*edof+ np.arange(NC*idof).reshape(NC, idof)
            def f1(bc):
                ps = mesh.bc_to_point(bc, etype='cell')
                return np.einsum('ijk, ijm->ijkm', u(ps), self.smspace.basis(ps, p=p-1))
            val = self.integralalg.cell_integral(f1, celltype=True)
            uh[cell2dof[:, 0:idof//2]] = val[:, 0, :] 
            uh[cell2dof[:, idof//2:]] = val[:, 1, :]
        return uh

    def mass_matrix(self):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        M = self.integralalg.construct_matrix(self.basis, cell2dof=cell2dof,
                gdof=gdof)
        return M

    def div_matrix(self):
        p = self.p
        gdof0 = self.number_of_global_dofs()
        cell2dof0 = self.cell_to_dof()
        gdof1 = self.smspace.number_of_global_dofs()
        cell2dof1 = self.smspace.cell_to_dof()
        basis0 = self.div_basis
        basis1 = lambda bc : self.smspace.basis(self.mesh.bc_to_point(bc), p=p)
        M = self.integralalg.construct_matrix(basis0, basis1=basis1, 
                cell2dof0=cell2dof0, gdof0=gdof0,
                cell2dof1=cell2dof1, gdof1=gdof1)
        return M

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
        from .femdof import multi_index_matrix2d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix2d(10)/10
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)

        if p == 0:
            m = 1
            n = 3
        elif p == 1:
            m = 4 
            n = 2 
        elif p == 2:
            m = 5 
            n = 3 
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], 
                    units='xy')

