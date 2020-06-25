import numpy as np
from numpy.linalg import inv

from .Function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

from ..decorator import barycentric # 导入默认的坐标类型, 这个空间是重心坐标

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
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None: #TODO: threshold 可以是一个指标数组
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None: #TODO: threshold 可以是一个指标数组
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
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return p*(p+1) 
        elif doftype in {'face', 'edge', 1}: # number of dofs on a edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        p = self.p

        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge') 
        gdof = NE*edof
        if p > 0:
            NC = self.mesh.number_of_cells()
            cdof = self.number_of_local_dofs(doftype='cell')
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
        # 单元上全部自由度的个数
        ldof = self.number_of_local_dofs(doftype='all')  

        cdof = self.smspace.number_of_local_dofs(doftype='cell')
        edof = self.smspace.number_of_local_dofs(doftype='edge')

        ndof = self.smspace.number_of_local_dofs(p=p) 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        
        LM, RM = self.smspace.edge_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        n = mesh.edge_unit_normal() 

        idx = self.smspace.edge_index_1(p=p+1)
        x = idx['x']
        y = idx['y']
        idx2 = np.arange(cdof)[None, None, :]
        idx3 = np.arange(GD*cdof, GD*cdof+edof)[None, None, :]

        # idx0 = edge2cell[:, [0]][:, None, None] this is a bug!!!
        idx0 = edge2cell[:, 0][:, None, None]
        idx1 = (edge2cell[:, [2]]*edof + np.arange(edof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*LM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  cdof+x] + n[:, 1, None, None]*LM[:, :, cdof+y]

        # idx0 = edge2cell[:, [1]][:, None, None] this is a bug!!!
        idx0 = edge2cell[:, 1][:, None, None]
        idx1 = (edge2cell[:, [3]]*edof + np.arange(edof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  cdof+x] + n[:, 1, None, None]*RM[:, :, cdof+y]

        if p > 0:
            M = self.smspace.cell_mass_matrix()
            idx = self.smspace.diff_index_1()
            idof = self.smspace.number_of_local_dofs(p=p-1, doftype='cell') 
            for i, key in enumerate(idx.keys()):
                index = np.arange((GD+1)*edof + i*idof, (GD+1)*edof+ (i+1)*idof)[:, None]
                A[:, index, i*cdof + np.arange(cdof)] = M[:, :idof, :]
                A[:, index, GD*cdof + np.arange(edof)] = M[:,  idx[key][0], cdof-edof:]
        return inv(A)

    @barycentric
    def face_basis(self, bc, index=None, barycenter=True):
        return self.edge_basis(bc, index, barycenter)


    @barycentric
    def edge_basis(self, bc, index=None, barycenter=True):
        """
        """
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        edge2cell = mesh.ds.edge_to_cell()

        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, etype='edge', index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=edge2cell[index, 0]) # (NQ, NE, ndof)

        shape = ps.shape[:-1] + (edof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NE, edof, 2)

        idx0 = edge2cell[index, 0][:, None]
        idx2 = edge2cell[index[:, None], [2]]*edof + np.arange(edof)
        c = self.bcoefs[idx0, :, idx2].swapaxes(-1, -2) # (NE, ldof, edof) 
        idx = self.smspace.edge_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def basis(self, bc, index=None, barycenter=True):
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

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all') 

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()

        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, etype='cell', index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.edge_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def div_basis(self, bc, index=None, barycenter=True):
        p = self.p
        ldof = self.number_of_local_dofs('all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.grad_basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, )
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[:] += np.einsum('ijm, jmn->ijn', val[..., :cdof, i], c[:, i*cdof:(i+1)*cdof, :])
            phi[:] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key], i], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def grad_basis(self, bc):
        pass

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc, index=None):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof])
        return val

    @barycentric
    def div_value(self, uh, bc, index=None):
        dphi = self.div_basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, dphi, uh[cell2dof])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=None):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def project(self, u):
        return self.interpolation(u)

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

    def stiff_matrix(self):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        M = self.integralalg.construct_matrix(self.basis, cell2dof0=cell2dof,
                gdof0=gdof)
        return M

    def div_matrix(self):
        p = self.p
        gdof0 = self.number_of_global_dofs()
        cell2dof0 = self.cell_to_dof()
        gdof1 = self.smspace.number_of_global_dofs()
        cell2dof1 = self.smspace.cell_to_dof()
        basis0 = self.div_basis
        basis1 = lambda bc : self.smspace.basis(self.mesh.bc_to_point(bc), p=p)

        D = self.integralalg.construct_matrix(basis0, basis1=basis1, 
                cell2dof0=cell2dof0, gdof0=gdof0,
                cell2dof1=cell2dof1, gdof1=gdof1)
        return D 

    def source_vector(self, f, dim=None, barycenter=False):
        cell2dof = self.smspace.cell_to_dof()
        gdof = self.smspace.number_of_global_dofs()
        b = -self.integralalg.construct_vector(f, self.smspace.basis, cell2dof, 
                gdof=gdof, dim=dim, barycenter=barycenter) 
        return b

    def neumann_boundary_vector(self, g, threshold=None, q=None):
        """
        Parameters
        ----------

        Notes
        ----
            For mixed finite element method, the Dirichlet boundary condition of
            Poisson problem become Neumann boundary condition, and the Neumann
            boundary condtion become Dirichlet boundary condition.
        """
        p = self.p
        mesh = self.mesh
        edof = self.smspace.number_of_local_dofs(doftype='edge') 
        edge2cell = mesh.ds.edge_to_cell()
        edge2dof = self.dof.edge_to_dof() 

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]
        en = mesh.edge_unit_normal(index=index)
        phi = self.edge_basis(bcs, index=index) 

        ps = mesh.bc_to_point(bcs, etype='edge', index=index)
        val = -g(ps)
        measure = self.integralalg.edgemeasure[index]

        gdof = self.number_of_global_dofs()
        F = np.zeros(gdof, dtype=self.ftype)
        bb = np.einsum('i, ij, ijmk, jk, j->jm', ws, val, phi, en, measure, optimize=True)
        np.add.at(F, edge2dof[index], bb)
        return F 

    def set_dirichlet_bc(self, uh, g, threshold=None, q=None):
        """
        """
        p = self.p
        mesh = self.mesh
        edge2dof = self.dof.edge_to_dof() 

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        ps = mesh.bc_to_point(bcs, etype='edge', index=index)
        en = mesh.edge_unit_normal(index=index)
        val = -g(ps, en)
        phi = self.smspace.edge_basis(ps, index=index)

        measure = self.integralalg.edgemeasure[index]
        gdof = self.number_of_global_dofs()
        uh[edge2dof[index]] = np.einsum('i, ij, ijm, j->jm', ws, val, phi,
                measure, optimize=True)
        isDDof = np.zeros(gdof, dtype=np.bool_) 
        isDDof[edge2dof[index]] = True
        return isDDof

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

