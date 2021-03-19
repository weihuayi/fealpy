
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix

from .Function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

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

        """

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        edof = self.number_of_local_dofs(doftype='edge') # 每条边内部的自由度
        idx = np.arange(edof) + ndof*NN

    def cell_to_dof(self, threshold=None):
        """

        Notes
        -----
        获取每个单元元上的自由度全局编号。
        """

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()


        ldof = self.number_of_local_dofs(doftype='all')  # 单元上的所有自由度
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        start = 0
        c2d = cell2dof[:, start:] # 顶点上的自由度, 一共 9 个

        cell = mesh.entity('cell')
        ndof = self.number_of_local_dofs(doftype='node')
        idx = np.arange(ndof)
        for i in range(3):
            c2d[:, ndof*i:ndof*(i+1)] = ndof*cell[:, i, None]
            c2d[:, ndof*i:ndof*(i+1)] += idx

        start += 3*ndof
        c2d = cell2dof[:, start:] # 边上的自由度, 共 3*(l-1+l) 个

        cell2edge = mesh.ds.cell_to_edge()
        edof = self.number_of_local_dofs(doftype='edge') # 每条边内部的自由度
        idx = np.arange(edof) + ndof*NN
        for i in range(3):
            c2d[:, edof*i:edof*(i+1)] = edof*cell2edge[:, i, None] 
            c2d[:, edof*i:edof*(i+1)] += idx

        start += 3*edof
        c2d = cell2dof[:, start:]
        cdof = self.number_of_local_dofs(doftype='cell') # 每个单元内部的自由度
        
        c2d[:, :] = np.arange(NC)[:, None]*np.arange(cdof)
        idx = np.arange(cdof) + ndof*NN + edof*NE
        c2d += idx 
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
