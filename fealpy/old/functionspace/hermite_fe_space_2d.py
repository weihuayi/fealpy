import numpy as np

from typing import Optional, Union, Callable
from .Function import Function
from ..decorator import barycentric

class HermiteFESpace2d:

    def __init__(self, mesh, p:int = 3):
        """
        @brief 
        """
        self.mesh = mesh # 三角形网格
        self.p = p



class HFEDof2d:

    def __init__(self, mesh, p:int = 3):
        self.mesh = mesh
        self.p = p
        self.itype = mesh.itype


    def multi_index_matrix(self):
        TD = self.mesh.top_dimension()
        ldof = self.number_of_local_dofs(doftype='all')
        mi = np.zeros((ldof, TD+1), dtype=self.itype)
        mi[0, :] = [p, 0, 0]
        mi[1, :] = [p-1, 1, 0]
        mi[2, :] = [p-1, 0, 1]
        mi[3, :] = [0, p, 0]
        mi[4, :] = []




    def cell_to_dof(self):
        """
        @brief 
        """
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()

    def node_to_dof(self, index=np.s_[:]):
        NN = self.mesh.number_of_nodes()
        node2dof = np.arange(3*NN).reshape(NN, 3) 
        return node2dof[index]

    def edge_to_dof(self, index=np.s_[:]):
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        if p > 3:
            edge2dof = 3*NN + np.arange((p-3)*NE).reshape(NE, -1)
            return edge2dof

    face_to_dof = edge_to_dof

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def interpolation_points(self, index=np.s_[:]):
        """
        @brief 
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()

        cell = self.mesh.entity('cell')
        node = self.mesh.entity('node')
        GD = self.mesh.geo_dimension()
        gdof = self.number_of_global_dofs()
        ips = np.zeros((gdof, GD), dtype=node.dtype)


    def number_of_global_dofs(self):
        """
        @brief  
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        gdof = 3*NN + (p-3)*NE + (p-2)*(p-1)//2
        return gdof


    def number_of_local_dofs(self, doftype='all'):
        """
        @brief
        """
        p = self.p
        if doftype == 'all': # 全部自由度
            return (p+1)*(p+2)//2
        elif doftype in {'cell', 2}: # 注意只有单元内部的
            return (p-2)*(p-1)//2
        elif doftype in {'face', 'edge',  1}: # 注意只有边内部的
            return p - 3 
        elif doftype in {'node', 0}:
            return 3
