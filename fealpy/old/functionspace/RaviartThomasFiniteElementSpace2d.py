import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix

from .Function import Function
from .scaled_monomial_space_2d import ScaledMonomialSpace2d

# 导入默认的坐标类型, 这个空间基函数的相关计算，输入参数是重心坐标 
from ..decorator import barycentric 

class RTDof2d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TriangleMesh object
        p : the space order, p>=0

        Notes
        -----

        2020.07.21 
        1. 这里不显式的存储 cell2dof, 需要的时候再构建，这样可以节约内存。
        2. 增加对存在裂缝网格区域的自由度管理支持。

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 0
        if 'fracture' in mesh.meshdata: # 此时存储的是 fracture 边的编号
            self.fracture = mesh.meshdata['fracture'] 
        elif 'fracture' in mesh.edgedata: # 此时存储的是 fracture 边的标记
            self.fracture, = np.nonzero(mesh.edgedata['fracture'])
        else:
            self.fracture = None

    @property
    def cell2dof(self):
        """
        
        Notes
        -----
        把这个方法属性化，保证老的程序接口不会出问题
        """
        return self.cell_to_dof()

    def boundary_dof(self, threshold=None):
        """
        """
        return self.is_boundary_dof(threshold=threshold)


    def is_boundary_dof(self, threshold=None):
        """

        Notes
        -----
        标记需要的边界自由度, 可用于边界条件处理。 threshold 用于处理混合边界条
        件的情形
        """

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        if threshold is None:
            flag = self.mesh.ds.boundary_edge_flag() # 全部的边界边编号
            edge2dof = self.edge_to_dof(threshold=flag)
        elif type(threshold) is np.ndarray: 
            edge2dof = self.edge_to_dof(threshold=threshold)
        elif callable(threshold):
            index = self.mesh.ds.boundary_edge_index()
            bc = self.mesh.entity_barycenter('edge', index=index)
            index = index[threshold(bc)]
            edge2dof = self.edge_to_dof(threshold=index)
        isBdDof[edge2dof] = True
        return isBdDof

    def edge_to_dof(self, threshold=None, doftype='left'):
        """

        Notes
        -----

        2020.07.21：
        获取网格边上的自由度全局编号。

        如果 threshold 不是 None 的话，则只获取一部分边上的自由度全局编号，这部
        分边由 threshold 来决定。

        left 参数用于网格中存在 fracture 边的情形。如果 doftype 为 'left'，则返回
        所有边左边单元上的全局自由度，如果为 'right'，则返回所有边右边单元上的全局
        自由度。注意对于 fracture 边，右边单元在这条边上自由度要多出一些。
        """
        mesh = self.mesh
        edof = self.number_of_local_dofs(doftype='edge')
        if threshold is None: # 所有的边上的自由度
            NE = mesh.number_of_edges()
            edge2dof = np.arange(NE*edof).reshape(NE, edof)
            if (doftype == 'right') and (self.fracture is not None):
                # 右边单元，并且存在 fracture 的情形
                NE = mesh.number_of_edges()
                NC = mesh.number_of_cells()
                cdof = self.number_of_local_dofs(doftype='cell')
                gdof0 = NE*edof + NC*cdof
                NFE = len(self.fracture) # 裂缝边的条数
                edge2dof[self.fracture] = np.arange(gdof0, gdof0 + NFE*edof, dtype=mesh.itype).reshape(NFE, edof)
            return edge2dof
        else: # 只获取一部分边上的自由度, 例如在混合边界条件的情形下，你只需要拿部分边界边
            if type(threshold) is np.ndarray: 
                if threshold.dtype == np.bool_:
                    index, = np.nonzero(threshold)
                else: # 否则为整数编号 
                    index = threshold
            elif callable(threshold):
                bc = self.mesh.entity_barycenter('edge')
                index, = np.nonzero(threshold(bc))
            edge2dof = edof*index.reshape(-1, 1) + np.arange(edof)
            return edge2dof

    def cell_to_dof(self, threshold=None):
        """

        Notes
        -----
        获取每个单元元上的自由度全局编号。
        """
        p = self.p 
        mesh = self.mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')  # 单元上的所有自由度
        cdof = self.number_of_local_dofs(doftype='cell') # 单元内部的自由度
        edof = self.number_of_local_dofs(doftype='edge') # 边内部的自由度
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        edge2dof0 = self.edge_to_dof(doftype='left')
        edge2dof1 = self.edge_to_dof(doftype='right')
        edge2cell = mesh.ds.edge_to_cell()
        cell2dof[edge2cell[:, [0]], edge2cell[:, [2]]*edof + np.arange(edof)] = edge2dof0 
        cell2dof[edge2cell[:, [1]], edge2cell[:, [3]]*edof + np.arange(edof)] = edge2dof1
        cell2dof[:, 3*edof:] = NE*edof+ np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+3) 
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return p*(p+1) 
        elif doftype in {'face', 'edge', 1}: # number of dofs on each edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        p = self.p
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        edof = self.number_of_local_dofs(doftype='edge') 
        cdof = self.number_of_local_dofs(doftype='cell')
        gdof = NE*edof + NC*cdof
        if self.fracture is not None:
            gdof += len(self.fracture)*edof
        return gdof 

class RaviartThomasFiniteElementSpace2d:
    """

    TODO
    ----
    """
    def __init__(self, mesh, p=0, q=None, dof=None):
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
    def face_basis(self, bc, index=np.s_[:], barycentric=True):
        return self.edge_basis(bc, index, barycentric)


    @barycentric
    def edge_basis(self, bc, index=np.s_[:], barycentric=True, left=True):
        """

        Notes
        -----
        计算每条边左边单元上的基函数， 在该边上的取值
        """
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        edge2cell = mesh.ds.edge_to_cell()

        if barycentric:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc

        if left:
            val = self.smspace.basis(ps, p=p+1, index=edge2cell[index, 0]) # (NQ, NE, ndof)
        else:
            val = self.smspace.basis(ps, p=p+1, index=edge2cell[index, 1]) # (NQ, NE, ndof)

        shape = ps.shape[:-1] + (edof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NE, edof, 2)

        if left:
            idx0 = edge2cell[index, 0][:, None]
            idx2 = edge2cell[index, 2][:, None]*edof + np.arange(edof)
        else:
            idx0 = edge2cell[index, 1][:, None]
            idx2 = edge2cell[index, 3][:, None]*edof + np.arange(edof)

        c = self.bcoefs[idx0, :, idx2].swapaxes(-1, -2) # (NE, ldof, edof) 
        idx = self.smspace.edge_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def basis(self, bc, index=np.s_[:], barycentric=True):
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

        if barycentric:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.edge_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('...jm, jmn->...jn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('...jm, jmn->...jn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def div_basis(self, bc, index=np.s_[:], barycentric=True):
        p = self.p
        ldof = self.number_of_local_dofs('all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        if barycentric:
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
    def value(self, uh, bc, index=np.s_[:]):
        
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        phi = self.basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        edge2dof = self.dof.edge_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof]) # index? 
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dphi = self.div_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, dphi, uh[cell2dof])# index ?
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def edge_value(self, uh, bc, index=np.s_[:]):
        phi = self.edge_basis(bc, index=index)
        edge2dof = self.dof.edge_to_dof() 
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    @barycentric
    def face_value(self, uh, bc, index=np.s_[:]):
        phi = self.edge_basis(bc, index=index)
        edge2dof = self.dof.edge_to_dof() 
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='barycentric')
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
            ps = mesh.bc_to_point(bc)
            return np.einsum('ijk, jk, ijm->ijm', u(ps), en, self.smspace.edge_basis(ps))
        uh[edge2dof] = self.integralalg.edge_integral(f0, edgetype=True)

        if p >= 1:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            edof = self.number_of_local_dofs('edge')
            idof = self.number_of_local_dofs('cell') # dofs inside the cell 
            cell2dof = NE*edof+ np.arange(NC*idof).reshape(NC, idof)
            def f1(bc):
                ps = mesh.bc_to_point(bc)
                return np.einsum('ijk, ijm->ijkm', u(ps), self.smspace.basis(ps, p=p-1))
            val = self.integralalg.cell_integral(f1, celltype=True)
            uh[cell2dof[:, 0:idof//2]] = val[:, 0, :] 
            uh[cell2dof[:, idof//2:]] = val[:, 1, :]
        return uh

    def stiff_matrix(self, q=None):
        """

        Notes
        -----
            基函数对应的矩阵, 和 mass_matrix 是一样的功能。
        """
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, q=q)
        return A

    def mass_matrix(self, q=None):
        """

        Notes
        -----
            基函数对应的矩阵，和 stiff_matrix 是一样的矩阵。
        """
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, q=q)
        return A

    def div_matrix(self, q=None):
        """

        Notes
        -----
            (div v, p)
        """
        gdof0 = self.number_of_global_dofs()
        cell2dof0 = self.cell_to_dof()
        b0 = (self.div_basis, cell2dof0, gdof0)

        gdof1 = self.smspace.number_of_global_dofs()
        cell2dof1 = self.smspace.cell_to_dof()
        b1 = (self.smspace.basis, cell2dof1, gdof1)
        B = self.integralalg.serial_construct_matrix(b0, b1=b1, q=q)
        return B 

    def pressure_matrix(self, ch, q=None):
        """

        Notes
        ----
            物质在单元上流入流出，是单元压力变化的原因。

            ch 是要考虑的的 n 种物质的浓度

            c_0 = ch.index(0)
            c_1 = ch.index(1)
            c_2 = ch.index(2)
            .....
            c_{n-1} = ch.index(n-1)

            目前仅考虑最低次元的情形，

            sum_i V_i (\\nabla \\cdot (c_i v), w) = V_i < c_i v\cdot n, w >_{\partial K}, 

            其中 V_i 是混合物的第 i 个组分偏摩尔体积，现在设为 1.

            这里不用考虑是物质的流入和流出吗？ 

            注意这里的 n 是单元的外法线，不是边的定向法线。

        TODO
        ----
            1. 考虑变化的 V_i 
        """

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)

        # 边的定向法线，它是左边单元的外法线， 右边单元内法线。
        en = mesh.edge_unit_normal() 
        measure = self.integralalg.edgemeasure # 边界长度

        # 边的左边单元在这条边上的 n 种成分的浓度值, (NQ, NE, ...) 
        val0 = ch(ps, index=edge2cell[:, 0]) 
        # 边的右边单元在这条边上的 n 种成分的浓度值, (NQ, NE, ...)
        val1 = ch(ps, index=edge2cell[:, 1])  
        if len(ch.shape) > 1:
            # TODO：考虑乘以偏摩尔体积系数
            val0 = np.sum(val0, axis=-1)
            val1 = np.sum(val1, axis=-1)

        # 压力空间左右单元基函数在共用边上的值
        # (NQ, NE, ldof0)
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps, index=edge2cell[:, 1])

        # (NQ, NE, ldof1, 2)
        phi2 = self.basis(ps, index=edge2cell[:, 0], barycentric=False)
        # (NQ, NE, ldof1)
        phi2 = np.einsum('...jln, jn->...jl', phi2, en)

        E0 = np.einsum(
                'i, ij..., ijm, ijn, j->jmn', 
                ws, val0, phi0, phi2, measure,
                optimize=True)
        E1 = np.einsum(
                'i, ij..., ijm, ijn, j->jmn', 
                ws, val1, phi1, phi2, measure,
                optimize=True)

        gdof0 = self.smspace.number_of_global_dofs()
        gdof1 = self.number_of_global_dofs()
        cell2dof = self.smspace.cell_to_dof()
        edge2dof = self.dof.edge_to_dof()

        I = np.broadcast_to(cell2dof[edge2cell[:, 0], :, None], shape=E0.shape)
        J = np.broadcast_to(edge2dof[:, None, :], shape=E0.shape)
        E = csr_matrix(
                (E0.flat, (I.flat, J.flat)), 
                shape=(gdof0, gdof1))

        I = np.broadcast_to(cell2dof[edge2cell[:, 1], :, None], shape=E1.shape)
        J = np.broadcast_to(edge2dof[:, None, :], shape=E0.shape)
        E += csr_matrix(
                (E1[isInEdge].flat, (I[isInEdge].flat, J[isInEdge].flat)),
                shape=(gdof0, gdof1))
        return E

    def source_vector(self, f, celltype=False, q=None):
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        b = (self.basis, cell2dof, gdof)
        F = self.integralalg.serial_construct_vector(f, b,
                celltype=celltype, q=q) 
        return F 

    def convection_vector(self, t, ch, vh, g=None, threshold=None, q=None):
        """

        Parameters
        ----------
        t: current time level
        ch: current concentration
        vh: current flow field
        g: boundary condition, g(x, t) = ch*vh \\cdot n 

        Notes
        -----
        (ch*vh, \\nabla p_h)_K - (ch*vh \\cdot n, p_h)_{\\partial K}

        注意单元内部浓度的变化要考虑从单元边界是流入， 还是流出， 流入导致单
        元内部物质浓度增加，流出导致物质浓度减少。

        Riemann Solvers
        Toro E. Riemann Solvers and Numerical Methods for Fluid dynamics. Springer: Berlin, 1997.
        """

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1]

        qf = self.integralalg.cellintegrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        measure = self.integralalg.cellmeasure
        ps = mesh.bc_to_point(bcs)
        val = vh(bcs) # TODO：考虑加入笛卡尔坐标的情形
        val *= ch(ps)[..., None]
        gphi = ch.space.grad_basis(ps) 

        F = np.einsum('i, ijm, ijkm, j->jk', ws, val, gphi, measure)

        # 边界积分
        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 边的定向法线，它是左边单元的外法线， 右边单元内法线。
        en = mesh.edge_unit_normal() 
        measure = self.integralalg.edgemeasure # 边界长度
        val0 = np.einsum('ijm, jm->ij', vh.edge_value(bcs), en) # 速度和法线的内积

        ps = mesh.bc_to_point(bcs)
        val1 = ch(ps, index=edge2cell[:, 0]) # 边的左边单元在这条边上的浓度值
        val2 = ch(ps, index=edge2cell[:, 1]) # 边的右边单元在这条边上的浓度值 

        # 边界条件处理
        val2[:, isBdEdge] = 0.0 # 首先把边界的贡献都设为 0 

        if g is not None:
            if type(threshold) is np.ndarray: # 下面考虑非零边界的贡献
                index = threshold # 这里假设 threshold 是边界边编号数组
            else:
                index = self.mesh.ds.boundary_edge_index()
                if threshold is not None:
                    bc = self.mesh.entity_barycenter('edge', index=index)
                    flag = threshold(bc)
                    index = index[flag]
            val2[:, index] = g(ps[:, index]) # 这里假设 g 是一个函数， TODO：其它情形？

        flag = val0 >= 0.0 # 对于左边单元来说，是流出项
                           # 对于右边单元来说，是流入项
        val = np.zeros_like(val0)  
        val[flag] = val0[flag]*val1[flag] 
        val[~flag] = val0[~flag]*val2[~flag]


        phi = ch.space.basis(ps, index=edge2cell[:, 0])
        b = np.einsum('i, ij, ijk, j->jk', ws, val, phi, measure)
        np.subtract.at(F, (edge2cell[:, 0], np.s_[:]), b)  

        phi = ch.space.basis(ps, index=edge2cell[:, 1])
        b = np.einsum('i, ij, ijk, j->jk', ws, val, phi, measure)
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1]) # 只处理内部边
        np.add.at(F, (edge2cell[isInEdge, 1], np.s_[:]), b[isInEdge])  

        return F


    def set_neumann_bc(self, g, threshold=None, q=None):
        """
        Parameters
        ----------

        Notes
        ----
        用混合有限元方法求解 Poisson 方程， Dirichlet 边界变为 Neumann 边界，
        Neumann 边界变化 Dirichlet 边界
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

        ps = mesh.bc_to_point(bcs, index=index)
        val = g(ps)
        if type(val) in {int, float}:
            val = np.array([[val]], dtype=self.ftype)
        measure = self.integralalg.edgemeasure[index]

        gdof = self.number_of_global_dofs()
        F = np.zeros(gdof, dtype=self.ftype)
        bb = np.einsum('i, ij, ijmk, jk, j->jm', ws, val, phi, en, measure, optimize=True)
        np.add.at(F, edge2dof[index], bb)
        return F 

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """

        Notes
        -----

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

        ps = mesh.bc_to_point(bcs, index=index)
        en = mesh.edge_unit_normal(index=index)
        val = gD(ps, en) # 注意这里容易出错
        if type(val) in {int, float}:
            val = np.array([[val]], dtype=self.ftype)
        phi = self.smspace.edge_basis(ps, index=index)

        measure = self.integralalg.edgemeasure[index]
        gdof = self.number_of_global_dofs()
        uh[edge2dof[index]] = np.einsum('i, ij, ijm, j->jm', ws, val, phi, measure, optimize=True)
        isDDof = np.zeros(gdof, dtype=np.bool_) 
        isDDof[edge2dof[index]] = True
        return isDDof

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def dof_array(self, dim=None):
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

