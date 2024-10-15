import numpy as np
from numpy.linalg import inv

from .Function import Function
from .scaled_monomial_space_3d import ScaledMonomialSpace3d

from ..decorator import barycentric # 导入默认的坐标类型, 这个空间是重心坐标

class RTDof3d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TetrahedronMesh or HalfEdgeMesh3d object
        p : the space order, p>=0

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 0
        self.itype = mesh.itype

        if 'fracture' in mesh.meshdata: # 此时存储的是 fracture 边的编号
            self.fracture = mesh.meshdata['fracture'] 
        elif 'fracture' in mesh.facedata: # 此时存储的是 fracture 边的标记
            self.fracture, = np.nonzero(mesh.facedata['fracture'])
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
            flag = self.mesh.ds.boundary_face_flag() # 全部的边界边编号
            face2dof = self.face_to_dof(threshold=flag)
        elif type(threshold) is np.ndarray: 
            face2dof = self.face_to_dof(threshold=threshold)
        elif callable(threshold):
            index = self.mesh.ds.boundary_edge_index()
            bc = self.mesh.entity_barycenter('face', index=index)
            index = index[threshold(bc)]
            edge2dof = self.face_to_dof(threshold=index)
        isBdDof[face2dof] = True
        return isBdDof

    def face_to_dof(self, threshold=None, doftype='left'):
        """

        Notes
        -----

        2020.07.21：
        获取网格面上的自由度全局编号。

        如果 threshold 不是 None 的话，则只获取一部分边上的自由度全局编号，这部
        分边由 threshold 来决定。

        left 参数用于网格中存在 fracture 边的情形。如果 doftype 为 'left'，则返回
        所有面左边单元上的全局自由度，如果为 'right'，则返回所有面右边单元上的全局
        自由度。注意对于 fracture 面，右边单元在这条边上自由度要多出一些。
        """
        mesh = self.mesh
        fdof = self.number_of_local_dofs(doftype='face')
        if threshold is None: # 所有的边上的自由度
            NF = mesh.number_of_faces()
            face2dof = np.arange(NF*fdof).reshape(NF, fdof)
            if (doftype == 'right') and (self.fracture is not None):
                # 右边单元，并且存在 fracture 的情形
                NF = mesh.number_of_faces()
                NC = mesh.number_of_cells()
                cdof = self.number_of_local_dofs(doftype='cell')
                gdof0 = NF*fdof + NC*cdof
                NFF = len(self.fracture) # 裂缝边的条数
                face2dof[self.fracture] = np.arange(gdof0, gdof0 + NFF*fdof,
                        dtype=mesh.itype).reshape(NFF, fdof)
            return face2dof
        else: # 只获取一部分边上的自由度, 例如在混合边界条件的情形下，你只需要拿部分边界边
            if type(threshold) is np.ndarray: 
                if threshold.dtype == np.bool_:
                    index, = np.nonzero(threshold)
                else: # 否则为整数编号 
                    index = threshold
            elif callable(threshold):
                bc = self.mesh.entity_barycenter('face')
                index, = np.nonzero(threshold(bc))
            face2dof = fdof*index.reshape(-1, 1) + np.arange(fdof)
            return face2dof

    def cell_to_dof(self):
        """
        """
        p = self.p 
        mesh = self.mesh

        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.number_of_local_dofs(doftype='cell') # 单元内部的自由度个数
        fdof = self.number_of_local_dofs(doftype='face') # 面上的自由度个数
        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        cell2dof = np.zeros((NC, ldof), dtype=self.itype)

        face2dof0 = self.face_to_dof(doftype='left')
        face2dof1 = self.face_to_dof(doftype='right')
        face2cell = mesh.ds.face_to_cell()

        cell2dof[face2cell[:, [0]], face2cell[:, [2]]*fdof + np.arange(fdof)] = face2dof0
        cell2dof[face2cell[:, [1]], face2cell[:, [3]]*fdof + np.arange(fdof)] = face2dof0

        cell2dof[:, 4*fdof:] = NF*fdof+ np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all':
            return (p+1)*(p+2)*(p+4)//2 
        elif doftype in {'cell', 3}:
            return p*(p + 1)*(p + 2)//2 
        elif doftype in {'face', 2}:
            return (p+1)*(p+2)//2
        else:
            return 0 

    def number_of_global_dofs(self):
        p = self.p
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        fdof = self.number_of_local_dofs(doftype='face') 
        cdof = self.number_of_local_dofs(doftype='cell')
        gdof = NF*fdof + NC*cdof
        if self.fracture is not None:
            gdof += len(self.fracture)*fdof
        return gdof 

class RaviartThomasFiniteElementSpace3d:
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
        RT_p : [P_{p-1}]^d(T) + [m_1, m_2, m_3]^T \\bar P_{p-1}(T)

        """
        self.p = p
        self.mesh = mesh
        self.smspace = ScaledMonomialSpace3d(mesh, p, q=q)

        if dof is None:
            self.dof = RTDof3d(mesh, p)
        else:
            self.dof = dof

        self.integralalg = self.smspace.integralalg
        self.integrator = self.smspace.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

        self.bcoefs = self.basis_coefficients()

    def basis_coefficients(self):
        """

        Parameters
        ----------

        Notes
        -----

        """
        p = self.p
        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell') 
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face')

        mesh = self.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        LM, RM = self.smspace.face_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        face = mesh.entity('face')
        face2cell = mesh.ds.face_to_cell()
        n = mesh.face_unit_normal() 

        idx = self.smspace.face_index_1(p=p+1)
        x = idx['x']
        y = idx['y']
        z = idx['z']
        idx2 = np.arange(cdof)[None, None, :]
        idx3 = np.arange(GD*cdof, GD*cdof+fdof)[None, None, :]

        # left cell
        idx0 = face2cell[:, 0][:, None, None]
        idx1 = (face2cell[:, [2]]*fdof + np.arange(fdof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*LM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  cdof+x] + \
                n[:, 1, None, None]*LM[:, :, cdof+y] + \
                n[:, 2, None, None]*LM[:, :, cdof+z]

        # right cell
        idx0 = face2cell[:, 1][:, None, None]
        idx1 = (face2cell[:, [3]]*fdof + np.arange(fdof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  cdof+x] + \
                n[:, 1, None, None]*RM[:, :, cdof+y] + \
                n[:, 2, None, None]*RM[:, :, cdof+z]

        if p > 0:
            M = self.smspace.cell_mass_matrix()
            idx = self.smspace.diff_index_1()
            idof = self.smspace.number_of_local_dofs(p=p-1, doftype='cell') 
            for i, key in enumerate(idx.keys()):
                index = np.arange(4*fdof + i*idof, 4*fdof+ (i+1)*idof)[:, None]
                A[:, index, i*cdof + np.arange(cdof)] = M[:, :idof, :]
                A[:, index, GD*cdof + np.arange(fdof)] = M[:,  idx[key][0], cdof-fdof:]
        return inv(A)

    @barycentric
    def face_basis(self, bc, index=np.s_[:], barycentric=True):
        """

        Paramerters
        -----------

        Notes
        -----
            给定三角形面中的一组重心坐标或者欧氏坐标点，计算这个三角形面对应的基函数在
            这些点处的函数值。
        """
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        face2cell = mesh.ds.face_to_cell()

        if barycentric:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=face2cell[index, 0]) # (NQ, NF, ndof)

        shape = ps.shape[:-1] + (fdof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NF, fdof, 2)

        idx0 = face2cell[index, 0][:, None]
        idx2 = face2cell[index, 2][:, None]*fdof + np.arange(fdof)
        c = self.bcoefs[idx0, :, idx2].swapaxes(-1, -2) # (NF, ldof, edof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def basis(self, bc, index=np.s_[:], barycentric=True):
        """

        Notes
        -----

        """
        p = self.p

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all') 

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

        mesh = self.mesh
        GD = mesh.geo_dimension()

        if barycentric:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, 3)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 3)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('...jm, jmn->...jn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('...jm, jmn->...jn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def div_basis(self, bc, index=np.s_[:], barycentric=True):
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

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
            phi[:] += np.einsum('...jm, jmn->...jn', val[..., :cdof, i], c[:, i*cdof:(i+1)*cdof, :])
            phi[:] += np.einsum('...jm, jmn->...jn', val[..., cdof+idx[key], i], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def grad_basis(self, bc):
        pass

    @barycentric
    def face_value(self, uh, bc, index=np.s_[:]):
        phi = self.face_basis(bc, index=index)
        face2dof = self.dof.face_to_dof() 
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[face2dof])
        return val

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof]) # TODO: index
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dphi = self.div_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, dphi, uh[cell2dof]) # TODO: index
        return val

    def project(self, u):
        return self.interpolation(u)

    def interpolation(self, u):
        p = self.p
        mesh = self.mesh

        uh = self.function()
        face2dof = self.dof.face_to_dof() 
        n = mesh.face_unit_normal()
        def f0(bc):
            ps = mesh.bc_to_point(bc)
            return np.einsum('ijk, jk, ijm->ijm', u(ps), n, self.smspace.face_basis(ps))
        uh[face2dof] = self.integralalg.face_integral(f0, edgetype=True)

        if p >= 1:
            NF = mesh.number_of_faces()
            NC = mesh.number_of_cells()
            fdof = self.number_of_local_dofs('face')
            idof = self.number_of_local_dofs('cell') # dofs inside the cell 
            cell2dof = NF*fdof+ np.arange(NC*idof).reshape(NC, idof)
            def f1(bc):
                ps = mesh.bc_to_point(bc)
                return np.einsum('ijk, ijm->ijkm', u(ps), self.smspace.basis(ps, p=p-1))
            val = self.integralalg.cell_integral(f1, celltype=True)
            idof = idof//3 
            uh[cell2dof[:, 0*idof:1*idof]] = val[:, 0, :] 
            uh[cell2dof[:, 1*idof:2*idof]] = val[:, 1, :]
            uh[cell2dof[:, 2*idof:3*idof]] = val[:, 2, :]
        return uh

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)


    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def stiff_matrix(self, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, q=q)
        return A

    def div_matrix(self, q=None):
        gdof0 = self.number_of_global_dofs()
        cell2dof0 = self.cell_to_dof()
        b0 = (self.div_basis, cell2dof0, gdof0)

        gdof1 = self.smspace.number_of_global_dofs()
        cell2dof1 = self.smspace.cell_to_dof()
        b1 = (self.smspace.basis, cell2dof1, gdof1)
        B = self.integralalg.serial_construct_matrix(b0, b1=b1, q=q)
        return B 

    def source_vector(self, f, celltype=False, q=None):
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        b = (self.basis, cell2dof, gdof)
        F = self.integralalg.serial_construct_vector(f, b,
                celltype=celltype, q=q) 
        return F 

    def set_neumann_bc(self, g, threshold=None, q=None):
        """
        Parameters
        ----------

        Notes
        ----
        """
        p = self.p
        mesh = self.mesh

        fdof = self.smspace.number_of_local_dofs(doftype='face') 
        face2cell = mesh.ds.face_to_cell()
        face2dof = self.dof.face_to_dof() 

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        fn = mesh.face_unit_normal(index=index)
        phi = self.face_basis(bcs, index=index) 

        ps = mesh.bc_to_point(bcs, index=index)
        val = g(ps)
        measure = self.integralalg.facemeasure[index]

        gdof = self.number_of_global_dofs()
        F = np.zeros(gdof, dtype=self.ftype)
        bb = np.einsum('i, ij, ijmk, jk, j->jm', ws, val, phi, fn, measure, optimize=True)
        np.add.at(F, face2dof[index], bb)
        return F 

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """

        Parameters
        ----------

        Notes
        -----
        """
        p = self.p
        mesh = self.mesh
        face2dof = self.dof.face_to_dof() 

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        ps = mesh.bc_to_point(bcs, index=index)
        fn = mesh.face_unit_normal(index=index)
        val = gD(ps, fn)
        phi = self.smspace.face_basis(ps, index=index)

        measure = self.integralalg.facemeasure[index]
        gdof = self.number_of_global_dofs()
        uh[face2dof[index]] = np.einsum('i, ij, ijm, j->jm', ws, val, phi,
                measure, optimize=True)
        isDDof = np.zeros(gdof, dtype=np.bool_) 
        isDDof[face2dof[index]] = True
        return isDDof

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
        face2cell = mesh.ds.face_to_cell()
        isBdFace = face2cell[:, 0] == face2cell[:, 1]

        qf = self.integralalg.cellintegrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        measure = self.integralalg.cellmeasure
        ps = mesh.bc_to_point(bcs)
        val = vh(bcs) # TODO：考虑加入笛卡尔坐标的情形
        val *= ch(ps)[..., None]
        gphi = ch.space.grad_basis(ps) 

        F = np.einsum('i, ijm, ijkm, j->jk', ws, val, gphi, measure)

        # 边界积分
        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 面的定向法线，它是左边单元的外法线， 右边单元内法线。
        fn = mesh.face_unit_normal() 
        measure = self.integralalg.facemeasure # 边界长度
        val0 = np.einsum('ijm, jm->ij', vh.face_value(bcs), fn) # 速度和法线的内积

        ps = mesh.bc_to_point(bcs, etype='face')
        val1 = ch(ps, index=face2cell[:, 0]) # 面的左边单元在这条边上的浓度值
        val2 = ch(ps, index=face2cell[:, 1]) # 面的右边单元在这条边上的浓度值 

        # 边界条件处理
        val2[:, isBdFace] = 0.0 # 首先把边界的贡献都设为 0 

        if g is not None:
            if type(threshold) is np.ndarray: # 下面考虑非零边界的贡献
                index = threshold # 这里假设 threshold 是边界边编号数组
            else:
                index = self.mesh.ds.boundary_face_index()
                if threshold is not None:
                    bc = self.mesh.entity_barycenter('face', index=index)
                    flag = threshold(bc)
                    index = index[flag]
            val2[:, index] = g(ps[:, index]) # 这里假设 g 是一个函数， TODO：其它情形？

        flag = val0 >= 0.0 # 对于左边单元来说，是流出项
                           # 对于右边单元来说，是流入项
        val = np.zeros_like(val0)  
        val[flag] = val0[flag]*val1[flag] 
        val[~flag] = val0[~flag]*val2[~flag]

        phi = ch.space.basis(ps, index=face2cell[:, 0])
        b = np.einsum('i, ij, ijk, j->jk', ws, val, phi, measure)
        np.subtract.at(F, (face2cell[:, 0], np.s_[:]), b)  

        phi = ch.space.basis(ps, index=face2cell[:, 1])
        b = np.einsum('i, ij, ijk, j->jk', ws, val, phi, measure)
        isInFace = (face2cell[:, 0] != face2cell[:, 1]) # 只处理内部面
        np.add.at(F, (face2cell[isInFace, 1], np.s_[:]), b[isInFace])  

        return F

    def cell_to_dof(self):
        return self.dof.cell_to_dof()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def show_basis(self, fig, index=0):
        """
        Plot quvier graph for every basis in a fig object
        """
        from .femdof import multi_index_matrix3d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix3d(4)/4
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)
        if p == 0:
            m = 2
            n = 2
        elif p == 1:
            m = 5
            n = 3
        elif p == 2:
            m = 6
            n = 6
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1, projection='3d')
            mesh.add_plot(axes)
            node = ps[:, index, :]
            v = phi[:, index, i, :]
            l = np.max(np.sqrt(np.sum(v**2, axis=-1)))
            v /=l
            axes.quiver(
                    node[:, 0], node[:, 1], node[:, 2], 
                    v[:, 0], v[:, 1], v[:, 2], length=1)
