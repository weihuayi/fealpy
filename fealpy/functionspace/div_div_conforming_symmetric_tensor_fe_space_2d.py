
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix

from .Function import Function
from .scaled_monomial_space_2d import ScaledMonomialSpace2d
from ..quadrature import GaussLegendreQuadrature

# 导入默认的坐标类型, 这个空间基函数的相关计算，输入参数是重心坐标 
from ..decorator import barycentric 

class DDCSTDof2d:
    def __init__(self, mesh, p=(2, 3)):
        """
        Parameters
        ----------
        mesh : TriangleMesh object
        p : the space order, p=(l, k), l >= k -1, k >= 3

        Notes
        -----


        Reference
        ---------
        """

        self.mesh = mesh
        self.p = p 

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

    def edge_to_dof(self, threshold=None):
        """

        Notes
        -----

        生成每个边上的自由度全局编号
        """

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        ndof = self.number_of_local_dofs(doftype='node') # 每个点上的自由度个数
        edof = self.number_of_local_dofs(doftype='edge') # 每条边内部的自由个数
        start = ndof*NN
        if threshold is None: # 所有的边上的自由度
            NE = mesh.number_of_edges()
            edge2dof = np.arange(start, start+NE*edof).reshape(NE, edof)
            return edge2dof
        else: # 只获取一部分边上的自由度
            if type(threshold) is np.ndarray: 
                if threshold.dtype == np.bool_:
                    index, = np.nonzero(threshold)
                else: # 否则为整数编号 
                    index = threshold
            elif callable(threshold):
                bc = self.mesh.entity_barycenter('edge')
                index, = np.nonzero(threshold(bc))
            edge2dof = edof*index.reshape(-1, 1) + np.arange(start, start+edof)
            return edge2dof


    def cell_to_dof(self, threshold=None):
        """

        Notes
        -----
            获取每个单元上自由度的全局编号。

            下面代码中的 c2d 是 cell2dof 数组的视图数组, 修改 c2d, 就是修改 
            cell2dof
        """

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()


        ldof = self.number_of_local_dofs(doftype='all') # 单元上的所有自由度个数
        cell2dof = np.zeros((NC, ldof), dtype=mesh.itype)

        start = 0
        c2d = cell2dof[:, start:] # 顶点自由度, 一共 9 个

        cell = mesh.entity('cell')
        ndof = self.number_of_local_dofs(doftype='node')
        idx = np.arange(ndof)
        for i in range(3):
            c2d[:, ndof*i:ndof*(i+1)] = ndof*cell[:, i, None]
            c2d[:, ndof*i:ndof*(i+1)] += idx

        start += 3*ndof
        # 边自由度, 共 3*(l-1+l) 个, 数组的视图
        c2d = cell2dof[:, start:] 

        cell2edge = mesh.ds.cell_to_edge()
        edof = self.number_of_local_dofs(doftype='edge') # 每条边内部的自由度
        idx = np.arange(edof) + ndof*NN
        for i in range(3):
            c2d[:, edof*i:edof*(i+1)] = edof*cell2edge[:, i, None] 
            c2d[:, edof*i:edof*(i+1)] += idx

        # 内部自由度, 共 (k-1)*k/2 + (l-1)*l
        start += 3*edof 
        c2d = cell2dof[:, start:]
        cdof = self.number_of_local_dofs(doftype='cell') # 每个单元内部的自由度
        start = ndof*NN + edof*NE
        c2d[:] = np.arange(start, start+NC*edof).reshape(NC, cdof)

        return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        l, k = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return l**2 + 5*l + 3 + k*(k-1)//2 
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return (k-1)*k//2 - 3 + (l-1)*l 
        elif doftype in {'face', 'edge', 1}: # number of dofs on each edge 
            return l-1 + l 
        elif doftype in {'node', 0}: # number of dofs on each node
            return 3 

    def number_of_global_dofs(self):
        l, k = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        ndof = self.number_of_local_dofs(doftype='node')
        edof = self.number_of_local_dofs(doftype='edge') 
        cdof = self.number_of_local_dofs(doftype='cell')
        gdof = NN*ndof + NE*edof + NC*cdof
        return gdof 

class DivDivConformingSymmetricTensorFESpace2d:
    """

    TODO
    ----
    """
    def __init__(self, mesh, p=(2, 3), q=None, dof=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        p : the space order, p=(l, k), l >= k-1, k >= 3
        q : the index of quadrature fromula
        dof : the object for degree of freedom

        Note
        ----


        """
        self.mesh = mesh
        self.p = p # (l, k)
        
        # 基础缩放单项式空间
        self.smspace = ScaledMonomialSpace2d(mesh, max(p), q=q)

        if dof is None:
            self.dof = DDCSTDof2d(mesh, p)
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
            计算基函数的系数, 基函数的表达形式为 
            [C_l, C_k]c
            
        """
        mesh  = self.mesh
        ftype = mesh.ftype

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        smspace = self.smspace
        l, k = self.p

        # 单元上全部自由度的个数
        aldof = self.number_of_local_dofs(doftype='all')  

        # 系数矩阵
        A = np.zeros((NC, aldof, aldof), dtype=ftype)

        # 1. 3 个单元节点处共 9 个自由度
        ## 0 号点
        ldof = smspace.number_of_local_dofs(p=l+1)
        kdof = smspace.number_of_local_dofs(p=k-2)
        gphi = smspace.grad_basis(node[cell[:, 0]], p=l+1, scaled=False)
        # C_l(K;\mbS)
        A[:, 0, 0:ldof-2] = gphi[:, 2:, 1]
        A[:, 1, ldof-2:2*ldof-3] = gphi[:, 1:, 0]
        A[:, 2, 0:ldof-2] = -gphi[:, 2:, 0]/2.0
        A[:, 2, ldof-2:2*ldof-3] = -gphi[:, 1:, 1]/2.0
        # C_k^\oplus(K;\mbS)
        phi = smspace.basis(node[cell[:, 0]], p=k-2) # 每个节点在节点 0 处取值
        A[:, 0, 2*ldof-3:] = phi[:, 3, None]*phi # x**2 m_{k-2}
        A[:, 1, 2*ldof-3:] = phi[:, 5, None]*phi # y**2 m_{k-2}
        A[:, 2, 2*ldof-3:] = phi[:, 4, None]*phi # xy m_{k-2}


        ## 1 号点
        gphi = smspace.grad_basis(node[cell[:, 1]], p=l+1, scaled=False)
        # C_l(K;\mbS)
        A[:, 3, 0:ldof-2] = gphi[:, 2:, 1]
        A[:, 4, ldof-2:2*ldof-3] = gphi[:, 1:, 0]
        A[:, 5, 0:ldof-2] = -gphi[:, 2:, 0]/2.0
        A[:, 5, ldof-2:2*ldof-3] = -gphi[:, 1:, 1]/2.0
        # C_k^\oplus(K;\mbS)
        phi = smspace.basis(node[cell[:, 1]], p=k-2)
        A[:, 3, 2*ldof-3:] = phi[:, 3, None]*phi # x**2 m_{k-2}
        A[:, 4, 2*ldof-3:] = phi[:, 5, None]*phi # y**2 m_{k-2}
        A[:, 5, 2*ldof-3:] = phi[:, 4, None]*phi # xy m_{k-2}

        ## 2 号点
        gphi = smspace.grad_basis(node[cell[:, 2]], p=l+1, scaled=False)
        # C_l(K;\mbS)
        A[:, 6, 0:ldof-2] = gphi[:, 2:, 1]
        A[:, 7, ldof-2:2*ldof-3] = gphi[:, 1:, 0]
        A[:, 8, 0:ldof-2] = -gphi[:, 2:, 0]/2.0
        A[:, 8, ldof-2:2*ldof-3] = -gphi[:, 1:, 1]/2.0
        # C_k^\oplus(K;\mbS)
        phi = smspace.basis(node[cell[:, 2]], p=k-2)
        A[:, 6, 2*ldof-3:] = phi[:, 3, None]*phi # x**2 m_{k-2}
        A[:, 7, 2*ldof-3:] = phi[:, 5, None]*phi # y**2 m_{k-2}
        A[:, 8, 2*ldof-3:] = phi[:, 4, None]*phi # xy m_{k-2}

        # 2. 边上的自由度，
        # 每条边上有 l-1 + l 个自由度
        # 每条边需要在左右单元上组装矩阵
        # 每条边上需要组装 4 块矩阵, 分别是 
        # [[E_00, E_01]
        #  [E_10, E_11]]
        edge  = mesh.entity('edge')
        h = mesh.entity_measure('edge')
        n, t = mesh.edge_frame() # 每条边上的标架, n: 法线， t： 切向

        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = GaussLegendreQuadrature(l) # 第 l 个积分公式
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.edge_bc_to_point(bcs) 
        phi = self.smspace.edge_basis(ps, p=l-1) 


        # 左边单元
        E = np.zeros((NE, l-1+l, aldof), dtype=ftype)
        gphi = smspace.grad_basis(ps, index=edge2cell[:, 0], p=l+1,
                scaled=False) # 不除最终的 h, 相当于 h_KC_l  

        # 右边单元

        # 单元内部自由度，共有 （k - 1)k/2 - 3 + (l-1)l 个自由度
        start = 9 + 3*(2*l - 1) # 除去点和边上的自由度
        kdof = smspace.number_of_local_dofs(p=k-2) 
        ldof = smspace.number_of_local_dofs(p=l-1)

        if k > 3:
            bcs, ws = self.integrator.get_quadrature_points_and_weights()
            point = mesh.bc_to_point(bcs)
            shape = point.shape[:-1]+(kdof, 3)
            hphi = np.zeros(shape, dtype=mesh.ftype)
            phi = self.basis(point, index=index, p=k-4)
            idx = self.diff_index_2(p=k-2)
            hphi[..., idx['xx'][0], 0] = np.einsum('i, ...i->...i', idx['xx'][1], phi)
            hphi[..., idx['yy'][0], 1] = np.einsum('i, ...i->...i', idx['yy'][1], phi)
            hphi[..., idx['xy'][0], 2] = np.einsum('i, ...i->...i', idx['xy'][1], phi)








