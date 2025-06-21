"""


Notes
-----

Authors
-------
    Huayi Wei, Chunyu Chen, Xin Wang
"""

import time
import numpy as np
import gmsh

from typing import Union
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
import inspect

from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .mesh_base import Mesh, Plotable
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from ..common import DynamicArray
from .triangle_mesh import TriangleMesh


class HalfEdgeMesh2d(Mesh, Plotable):
    def __init__(self, node, halfedge, NC=None, NV=None, nodedof=None,
            initlevel=True):
        """

        Parameters
        ----------
        node : (NN, GD)
        halfedge : (2*NE, 4), 
            halfedge[i, 0]: the index of the vertex the i-th halfedge point to
            halfedge[i, 1]: the index of the cell the i-th halfedge blong to
            halfedge[i, 2]: the index of the next halfedge of i-th haledge 
            halfedge[i, 3]: the index of the prev halfedge of i-th haledge 
            halfedge[i, 4]: the index of the opposit halfedge of the i-th halfedge
        Notes
        -----
        这是一个用半边数据结构存储网格拓扑关系的类。半边数据结构表示的网格更适和
        网格的自适应算法的实现。

        这个类的核心数组都是动态数组， 可以根据网格实体数目的变化动态增加长度，
        理论上可有效减少内存开辟的次数。

        边界半边的对边是自身

        Reference
        ---------
        [1] https://github.com/maciejkula/dynarray/blob/master/dynarray/dynamic_array.py
        """

        self.itype = halfedge.dtype
        self.ftype = node.dtype
        
        self.meshtype = 'tri'
        self.type = "TRI"

        self.node = DynamicArray(node, dtype = node.dtype)
        self.ds = HalfEdgeMesh2dDataStructure(halfedge, NN = node.shape[0], NC=NC, NV=NV)
        self.meshtype = 'halfedge2d'

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.meshdata = {}
        self.halfedgedata = {}
        self.facedata = self.edgedata

        # 网格节点的自由度标记数组
        # 0: 固定点
        # 1: 边界上的点
        # 2: 区域内部的点
        if nodedof is not None:
            self.nodedata['dof'] = nodedof

        if initlevel:
            self.init_level_info()

        if NV==3:
            self.shape_function = self._shape_function
            self.cell_shape_function = self._shape_function
            self.face_shape_function = self._shape_function
            self.edge_shape_function = self._shape_function

    @classmethod
    def from_mesh(cls, mesh, NV=None):
        """

        Notes
        -----
        输入一个其它类型数据结构的网格，转化为半边数据结构。如果 closed 为真，则
        表明输入网格是一个封闭的曲面网格；为假，则为开的网格，可以存在洞，或者无
        界的外界区域
        """
        mtype = mesh.meshtype
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        if mtype not in {'halfedge', 'halfedge2d'}:
            NE = mesh.number_of_edges()
            NBE = mesh.ds.boundary_edge_flag().sum()
            NHE = NE*2 - NBE # 半边数, NHE - NBE 为内部边的个数的二倍

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()

            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
            isBdEdge = ~isInEdge

            idx = np.zeros((NHE, 2), dtype=np.int_)
            halfedge = np.zeros((NHE, 5), dtype=mesh.itype)
            halfedge[:NHE-NBE, 0] = edge[isInEdge].flat
            halfedge[NHE-NBE:, 0] = edge[isBdEdge, 1]

            halfedge[0:NHE-NBE:2, 1] = edge2cell[isInEdge, 1]
            halfedge[1:NHE-NBE:2, 1] = edge2cell[isInEdge, 0]
            halfedge[NHE-NBE:, 1] = edge2cell[isBdEdge, 0]

            halfedge[0:NHE-NBE:2, 4] = np.arange(1, NHE-NBE, 2)
            halfedge[1:NHE-NBE:2, 4] = np.arange(0, NHE-NBE, 2)
            halfedge[NHE-NBE:, 4] = np.arange(NHE-NBE, NHE) 

            idx[0:NHE-NBE:2, 1] = edge2cell[isInEdge, 3]
            idx[1:NHE-NBE:2, 1] = edge2cell[isInEdge, 2]
            idx[NHE-NBE:, 1] = edge2cell[isBdEdge, 2]
            idx[:, 0] = halfedge[:, 1]

            idx = np.lexsort([idx[:, 1], idx[:, 0]])
            halfedge[idx, 2] = np.roll(idx, -1)
            halfedge[idx, 3] = np.roll(idx, 1)

            idx0 = np.where(halfedge[halfedge[idx, 2], 1]!=halfedge[idx, 1])[0]
            idx1 = np.where(halfedge[halfedge[idx, 3], 1]!=halfedge[idx, 1])[0]
            halfedge[idx[idx0], 2] = idx[idx1]
            halfedge[idx[idx1], 3] = idx[idx0]

            mesh =  cls(node, halfedge, NC=NC, NV=NV)
            return mesh
        else:
            newMesh =  cls(mesh.node.copy(), mesh.ds.halfedge.copy(), NC=NC, NV=mesh.ds.NV)
            newMesh.celldata['level'][:] = mesh.celldata['level']
            newMesh.halfedge['level'][:] = mesh.halfedgedata['level']
            return newMesh

    def init_level_info(self):
        """
        @brief 初始化半边和单元的 level 
        """
        NN = self.number_of_nodes()
        NHE = self.ds.number_of_halfedges()
        NC = self.number_of_cells() # 实际单元个数

        self.halfedgedata['level'] = DynamicArray((NHE, ), val=0,dtype=np.int_)
        self.celldata['level'] = DynamicArray((NC, ), val=0, dtype=np.int_) 

        # 如果单元的角度大于 170 度， 设对应的半边层数为 1
        node = self.node
        halfedge = self.ds.halfedge
        v0 = node[halfedge[halfedge[:, 2], 0]] - node[halfedge[:, 0]]
        v1 = node[halfedge[halfedge[:, 3], 0]] - node[halfedge[:, 0]]
        angle = np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1)*np.sum(v1**2, axis=1))
        self.halfedgedata['level'][(angle < -0.98)] = 1 

    def convexity(self):#
        """
        @brief 将网格中的非凸单元分解为凸单元
        """
        def angle(x, y):
            x = x/(np.linalg.norm(x, axis=1)).reshape(len(x), 1)
            y = y/np.linalg.norm(y, axis=1).reshape(len(y), 1)
            theta = np.sign(np.cross(x, y))*np.arccos((x*y).sum(axis=1))
            theta[theta<0]+=2*np.pi
            theta[theta==0]+=np.pi
            return theta

        node = self.entity('node')
        halfedge = self.entity('halfedge')

        hedge = self.ds.hedge
        hcell = self.ds.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        while True:
            NC = self.number_of_cells()
            NHE = self.ds.number_of_halfedges()

            end = halfedge[:, 0]
            start = halfedge[halfedge[:, 3], 0]
            vector = node[end] - node[start]#所有边的方向
            vectornex = vector[halfedge[:, 2]]#所有边的下一个边的方向

            angle0 = angle(vectornex, -vector)#所有边的反方向与下一个边的角度
            badHEdge, = np.where(angle0 > np.pi)#夹角大于170度的半边
            badCell, idx= np.unique(halfedge[badHEdge, 1], return_index=True)#每个单元每次只处理中的一个
            badHEdge = badHEdge[idx]#现在坏半边的单元都不同
            badNode = halfedge[badHEdge, 0]
            NE1 = len(badHEdge)

            nex = halfedge[badHEdge, 2]
            pre = halfedge[badHEdge, 3]
            vectorBad = vector[badHEdge]#坏半边的方向
            vectorBadnex = vector[nex]#坏半边的下一个半边的方向

            anglenex = angle(vectorBadnex, -vectorBad)#坏边的夹角
            anglecur = anglenex/2#当前方向与角平分线的夹角
            angle_err_min = anglenex/2#与角平分线夹角的最小值
            goodHEdge = np.zeros(NE1, dtype=np.int_)#最小夹角的边
            isNotOK = np.ones(NE1, dtype = np.bool_)#每个单元的循环情况
            nex = halfedge[nex, 2]#从下下一个边开始
            while isNotOK.any():
                vectornex = node[halfedge[nex, 0]] - node[badNode]
                anglecur[isNotOK] = angle(vectorBadnex[isNotOK], vectornex[isNotOK])
                angle_err = abs(anglecur - anglenex/2)
                goodHEdge[angle_err<angle_err_min] = nex[angle_err<angle_err_min]#与角平分线夹角小于做小夹角的边做goodHEdge.
                angle_err_min[angle_err<angle_err_min] = angle_err[angle_err<angle_err_min]#更新最小角
                nex = halfedge[nex, 2]
                isNotOK[nex==pre] = False#循环到坏边的上上一个边结束
            halfedgeNew = halfedge.increase_size(NE1*2)
            halfedgeNew[:NE1, 0] = halfedge[goodHEdge, 0].copy()
            halfedgeNew[:NE1, 1] = halfedge[badHEdge, 1].copy()
            halfedgeNew[:NE1, 2] = halfedge[goodHEdge, 2].copy()
            halfedgeNew[:NE1, 3] = badHEdge.copy()
            halfedgeNew[:NE1, 4] = np.arange(NHE+NE1, NHE+NE1*2)

            halfedgeNew[NE1:, 0] = halfedge[badHEdge, 0].copy()
            halfedgeNew[NE1:, 1] = np.arange(NC, NC+NE1)
            halfedgeNew[NE1:, 2] = halfedge[badHEdge, 2].copy()
            halfedgeNew[NE1:, 3] = goodHEdge.copy()
            halfedgeNew[NE1:, 4] = np.arange(NHE, NHE+NE1)

            halfedge[halfedge[goodHEdge, 2], 3] = np.arange(NHE, NHE+NE1)
            halfedge[halfedge[badHEdge, 2], 3] = np.arange(NHE+NE1, NHE+NE1*2)
            halfedge[badHEdge, 2] = np.arange(NHE, NHE+NE1)
            halfedge[goodHEdge, 2] = np.arange(NHE+NE1, NHE+NE1*2)
            isNotOK = np.ones(NE1, dtype=np.bool_)
            nex = halfedge[len(halfedge)-NE1:, 2]
            while isNotOK.any():
                halfedge[nex[isNotOK], 1] = np.arange(NC, NC+NE1)[isNotOK]
                nex = halfedge[nex, 2]
                flag = (nex==np.arange(NHE+NE1, NHE+NE1*2)) & isNotOK
                isNotOK[flag] = False

            #单元层
            clevelNew = clevel.increase_size(NE1)
            clevelNew[:] = clevel[halfedge[badHEdge, 1]]

            #半边层
            hlevelNew = hlevel.increase_size(NE1*2)
            hlevelNew[:NE1] = hlevel[goodHEdge]
            hlevelNew[NE1:] = hlevel[badHEdge]

            self.ds.reinit()
            if len(badHEdge)==0:
                break

    def find_point(self, points, start=None): 
        """
        @brief 在网格中查找给定点的位置单元格。

        @param points numpy.ndarray 具有形状 (NP, 2) 的点数组，其中 NP 是点的数量。

        @param start numpy.ndarray 可选。具有形状 (NP,) 的整数数组，表示每个点的起始单元格索引。如果未提供，默认为零。

        @param return_lambda bool 可选。如果为 True，则返回每个点 lambda 值，否则返回单元格索引。默认为 False。

        @return numpy.ndarray 如果 return_lambda 为 True，则返回一个整数数组，表示每个点在所在单元的重心坐标。
        """
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hcell = self.ds.hcell

        cm = self.entity_measure('cell')

        NP = points.shape[0]
        start = np.zeros(NP, dtype=self.itype) if start is None else start
        cellstart = start.copy()

        isNotOK = np.ones(NP, dtype=np.bool_)
        while isNotOK.any():
            idx = np.where(isNotOK)[0]
            p1 = node[halfedge[start[isNotOK], 0]]
            p0 = node[halfedge[halfedge[start[isNotOK], 3], 0]]

            v0 = p1 - p0
            v1 = points[idx] - p0
            flag = np.cross(v0, v1)>0

            start[idx[flag]] = halfedge[start[idx[flag]], 2]
            cellstart[idx[~flag]] = halfedge[start[idx[~flag]], 4]
            start[idx[~flag]] = halfedge[halfedge[start[idx[~flag]], 4], 2]

            isNotOK[idx] = cellstart[idx]!=start[idx]

        return halfedge[start, 1]

########################### 三角形网格的接口 ####################################
    def find_point_in_triangle_mesh(self, points, start=None):
        """
        @brief 在三角形网格中查找给定点的位置单元格和重心坐标。

        @param points numpy.ndarray 具有形状 (NP, 2) 的点数组，其中 NP 是点的数量。

        @param start numpy.ndarray 可选。具有形状 (NP,) 的整数数组，表示每个点的起始单元格索引。如果未提供，默认为零。

        @return numpy.ndarray 具有形状 (NP,) 的整数数组，表示每个点的位置单元格索引。
                numpy.ndarray 具有形状 (NP, 3) 的浮点数数组，表示每个点的重心坐标（barycentric coordinates）。
        """

        # 获取网格的半边、节点和单元格信息
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        cell = self.entity('cell')
        cell = np.array(cell)
        hcell = self.ds.hcell

        # 计算每个单元格的测度（measure），并乘以2.0
        cm = self.entity_measure('cell')[:, None] * 2.0

        # 获取点的数量
        NP = points.shape[0]
        NC = hcell.shape[0]

        if start is None: 
            isBdNode = self.ds.boundary_node_flag()
            isBDCell = np.any(isBdNode[cell], axis=1)
            isNotBDCell = np.where(~isBDCell)[0]
            if self.ds.tree is None:
                self.tree = cKDTree(node[halfedge[hcell[isNotBDCell], 0]])
            start = isNotBDCell[self.tree.query(points, k=1)[1]]

        ## NP < NC//2 和 NP > NC//2, 出于效率的原因分成两个情况
        if NP < NC//2:
            # 创建用于存储计算结果的数组
            cell2hedge = np.zeros((NP, 3), dtype=np.int_)
            cell2vector = np.zeros((NP, 3, 2), dtype=np.float_)
            vector = np.zeros((NP, 3, 2), dtype=np.float_)
            bc = np.zeros((NP, 3), dtype=np.float_)

            # 创建用于跟踪哪些点尚未找到其位置单元格的布尔数组
            isNotOK = np.ones(NP, dtype=np.bool_)

            # 进入循环，直到所有点都找到了位置单元格
            k = 2
            while isNotOK.any():
                # 计算每个点所在的单元格到其相邻半边的映射
                cell2hedge[isNotOK, 0] = hcell[start[isNotOK]]
                cell2hedge[isNotOK, 1] = halfedge[hcell[start[isNotOK]], 2]
                cell2hedge[isNotOK, 2] = halfedge[hcell[start[isNotOK]], 3]

                # 计算每个点到其相邻半边的向量以及到目标点的向量
                cell2vector[isNotOK] = node[halfedge[cell2hedge[isNotOK], 0]] - node[
                        halfedge[halfedge[cell2hedge[isNotOK], 3], 0]]
                vector[isNotOK] = points[isNotOK][:, None] - node[
                        halfedge[halfedge[cell2hedge[isNotOK], 3], 0]]

                # 计算每个点的重心坐标（barycentric coordinates）
                bc[isNotOK] = np.cross(cell2vector[isNotOK], vector[isNotOK]) / cm[start[isNotOK]]

                # 找到重心坐标中小于零的点
                idx, jdx = np.where(bc[isNotOK] < -1e-12)
                index = np.where(isNotOK)[0][idx]

                # 更新未找到位置单元格的点的状态和起始单元格索引
                isNotOK[:] = False
                isNotOK[index] = True
                newstart = halfedge[halfedge[cell2hedge[index, jdx], 4], 1]
                isFaultPoint = index[newstart==start[index]]
                start[index] = newstart
                if len(isFaultPoint)>0:
                    start[isFaultPoint] = isNotBDCell[self.tree.query(points[isFaultPoint], k=k)[1][:, -1]]
                    k += 1

            return start, bc
        else:
            cell2hedge = np.zeros((NC, 3), dtype=np.int_)
            cell2hedge[:, 0] = hcell[:]
            cell2hedge[:, 1] = halfedge[hcell[:], 2]
            cell2hedge[:, 2] = halfedge[hcell[:], 3]

            # 计算每个点到其相邻半边的向量以及到目标点的向量
            #cell2vector = node[halfedge[cell2hedge, 0]] - node[
            #        halfedge[halfedge[cell2hedge, 3], 0]]
            cell2vector = node[cell[:, [2, 0, 1]]] - node[cell[:, [1, 2, 0]]]

            bc = np.zeros((NP, 3), dtype=np.float_)

            # 创建用于跟踪哪些点尚未找到其位置单元格的布尔数组
            isNotOK = np.ones(NP, dtype=np.bool_)
            k=2
            # 进入循环，直到所有点都找到了位置单元格
            while isNotOK.any():
                # 计算每个点到其相邻半边的向量以及到目标点的向量
                current = start[isNotOK]
                vector = points[isNotOK][:, None] - node[cell[current][:, [1, 2, 0]]]

                # 计算每个点的重心坐标（barycentric coordinates）
                bc[isNotOK] = np.cross(cell2vector[current], vector)/cm[current]

                # 找到重心坐标中小于零的点
                idx, jdx = np.where(bc[isNotOK] < -1e-12)
                index = np.where(isNotOK)[0][idx]

                # 更新未找到位置单元格的点的状态和起始单元格索引
                isNotOK[:] = False
                isNotOK[index] = True

                newstart = halfedge[halfedge[cell2hedge[current[idx], jdx], 4], 1]
                isFaultPoint = index[newstart==start[index]]
                start[index] = newstart
                if len(isFaultPoint)>0:
                    start[isFaultPoint] = isNotBDCell[self.tree.query(points[isFaultPoint], k=k)[1][:, -1]]
                    k += 1
            return start, bc

    def interpolation_cell_data(self, mesh, datakey, itype="max"):
        """
        @brief 在网格中进行单元数据的插值。

        @param
          mesh: 网格对象，包含了要进行插值的数据。
          datakey: 一个包含要插值的数据键的列表。
          itype: 插值类型，可以是 'max'（最大值）、'average'（平均值）或 'min'（最小值）之一。
        """
        NC = self.number_of_cells()
        fun = {'max' : np.max, 'average' : np.average, 'min' : np.min} # 定义插值函数（最大值、平均值和最小值）
        bcs = np.array([[1/12, 1/24, 7/8], [5/6, 1/24, 1/8],
                        [11/32, 5/12, 23/96], [1/24, 5/6, 1/8]]) # 定义四个插值点的重心坐标
        loc = None # 初始化loc
        celldata = {s: np.zeros((4, NC), dtype=mesh.celldata[s].dtype) for s in datakey} # 创建用于存储插值结果的字典
        for i in range(4): # 遍历四个插值点
            points = self.bc_to_point(bcs[i, None, :])[0] # 根据重心坐标计算插值点的实际坐标
            loc, _ = mesh.find_point_in_triangle_mesh(points, start=loc) # 在网格中查找最接近的点

            for key in datakey: # 针对每个数据键进行插值
                odata = mesh.celldata[key]
                celldata[key][i] = odata[loc]

        for key in datakey: # 根据插值类型计算插值结果
            self.celldata[key] = fun[itype](celldata[key], axis=0)


    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        if self.ds.NV==3:
            R = self._grad_shape_function(bc, p)
            if variables == 'x':
                Dlambda = self.grad_lambda(index=index)
                gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
                return gphi #(NQ, NC, ldof, GD)
            elif variables == 'u':
                return R #(NQ, ldof, TD+1)
        else:
            pass

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        """
        @brief 计算单元上所有形函数在边上的积分点处的导函数值

        @param bc 边上的一组积分点
        @param cindex 边所在的单元编号
        @param lidx 边在该单元的局部编号
        @param direction  True 表示边的方向和单元的逆时针方向一致，False 表示不一致
        """

        NC = len(cindex)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NC, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NC)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        gphi = self.grad_shape_function(bcs, p=p, index=cindex, variables='x')

        return gphi

    grad_shape_function_on_face = grad_shape_function_on_edge

    def line_to_cell(self, point, segment):
        """
        @brief Find the cells intersected by the line segment defined by the given points.

        @param point numpy.ndarray An array of points with shape (N, 2), 
            where N is the number of points.
        @param segment numpy.ndarray An array of segments with shape (M, 2), 
            where M is the number of segments.

        @return tuple A tuple containing two numpy arrays:
           - isCrossCell numpy.ndarray A boolean array of size NC, 
            indicating whether each cell is intersected by the line segment.
           - intersectedCells numpy.ndarray An array containing the indices of 
            cells that are intersected by the line segment.
        """
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hcell = self.ds.hcell
        NC = self.ds.number_of_cells()

        x = point[segment[:, 0]]
        y = point[segment[:, 1]]

        start = self.node_to_cell(x)
        end = self.node_to_cell(y)
        isCrossCell = np.zeros(NC, dtype=np.bool_)
        isCrossCell[start] = True
        vector = x-y

        while len(start)>0:
            cell2hedge = np.c_[hcell[start],
                    halfedge[hcell[start], 2], halfedge[hcell[start], 3]].T
            cell2vector = node[halfedge[cell2hedge, 0]] - y
            area = np.cross(cell2vector, vector)
            area0 = np.r_['0', np.array([area[-1]]), area[:-1]]
            flag = ((area>=0)&(area0<0)).T
            start = halfedge[halfedge[cell2hedge.T[flag], 4], 1]
            isCrossCell[start] = True
            flag = (start==end)
            start = start[~flag]
            end = end[~flag]
            y = y[~flag]
            vector = vector[~flag]
        return isCrossCell, np.where(isCrossCell)

############### 以下是模仿 PolygonMesh 的接口 #############################
    def integrator(self, q, etype='cell', qtype='legendre'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            if qtype in {'legendre'}:
                from ..quadrature import GaussLegendreQuadrature
                return GaussLegendreQuadrature(q)
            elif qtype in {'lobatto'}:
                from ..quadrature import GaussLobattoQuadrature
                return GaussLobattoQuadrature(q)

    def entity(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.ds.cell_to_node()[index]
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge_to_node()[index]
        elif etype in {'halfedge'}:
            return self.ds.halfedge # DynamicArray
        elif etype in {'node', 0}:
            return self.node # DynamicArrray
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell', index=np.s_[:]):
        node = self.entity('node')
        if etype in {'cell', 2}:
            cell2node = self.ds.cell_to_node(return_sparse=True)
            NV = self.ds.number_of_vertices_of_cells().reshape(-1, 1)
            bc = cell2node*node/NV
            return bc
        elif etype in {'edge', 'face', 1}:
            GD = self.geo_dimension()
            edge = self.ds.edge_to_node()[index]
            bc = np.sum(node[edge, :], axis=1).reshape(-1, GD)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node
        elif etype in {'halfedge'}:
            halfedge = self.entity('halfedge')
            bc = 0.5*(node[halfedge[index, 0]] + node[halfedge[halfedge[index, 3], 0]])
        return bc

    def get_internal_point(self):
        NC = self.number_of_cells()

        node = self.entity("node")
        halfedge = self.entity("halfedge")
        nex = halfedge[:, 2]
        pre = halfedge[:, 3]

        v = node[halfedge[pre, 0]] - node[halfedge[:, 0]]
        flag = np.cross(v, v[nex])>0

        point = (node[halfedge[flag, 0]]+
                 node[halfedge[nex[flag], 0]]+
                 node[halfedge[pre[flag], 0]])/3.0
        re = np.zeros((NC, node.shape[1]), dtype=np.float_)
        re[halfedge[flag, 1]] = point
        return re


    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.halfedge_length()[self.ds.hedge][index]
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def edge_bc_to_point(self, bcs, index=np.s_[:]):
        """
        @brief 重心坐标积分点转化为网格边上笛卡尔坐标点
        """
        node = self.entity('node')
        edge = self.entity('edge')[index]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        return ps

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 
        """
        node = self.entity('node')
        if self.ds.NV == 3:
            TD = bc.shape[-1] - 1 # bc.shape == (NQ, TD+1)
            node = self.entity('node')
            entity = self.entity(etype=TD)[index] # default  cell
            p = np.einsum('...j, ijk->...ik', bc, node[entity])
        elif self.ds.NV == 4:
            if isinstance(bc, tuple):
                assert len(bc) == 2
                cell = self.entity('cell')[index]
                bc0 = bc[0] # (NQ0, 2)
                bc1 = bc[1] # (NQ1, 2)
                bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)
                p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
            else:
                edge = self.entity('edge')[index]
                p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        elif self.ds.NV is None:
            TD = bc.shape[-1] - 1 # bc.shape == (NQ, TD+1)
            node = self.entity('node')
            entity = self.entity(etype=TD)[index] # default  cell
            p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief 插值点总数
        """
        return self.ds.number_of_global_ipoints(p)

    def number_of_local_ipoints(self,
            p: int, iptype: Union[int, str]='all') -> Union[NDArray, int]:
        """
        @brief 获取局部插值点的个数
        """
        return self.ds.number_of_local_ipoints(p, iptype)

    def cell_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief
        """
        return self.ds.cell_to_ipoint(p, index)

    def edge_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief 获取网格边与插值点的对应关系

        """
        return self.ds.edge_to_ipoint(p, index)

    face_to_ipoint = edge_to_ipoint

    def geo_dimension(self):
        return self.node.shape[1]
    
    def node_normal(self):
        node = self.node
        cell = self.entity('cell')
        if isinstance(cell, tuple):
            cell, cellLocation = cell
            idx1 = np.zeros(cell.shape[0], dtype=np.int)
            idx2 = np.zeros(cell.shape[0], dtype=np.int)

            idx1[0:-1] = cell[1:]
            idx1[cellLocation[1:]-1] = cell[cellLocation[:-1]]
            idx2[1:] = cell[0:-1]
            idx2[cellLocation[:-1]] = cell[cellLocation[1:]-1]
            w = np.array([(0,-1),(1,0)])
            d = node[idx1] - node[idx2]
            return 0.5*d@w
        else:
            assert self.ds.NV == 3 or self.ds.NV == 4
            # TODO: for tri and quad case

    def cell_area(self, index=np.s_[:]):
        """
        @brief 计算单元的面积
        """
        if self.ds.NV in {None, 4}:
            NC = self.number_of_cells()
            node = self.entity('node')

            halfedge = self.ds.halfedge # DynamicArray

            e0 = halfedge[halfedge[:, 3], 0]
            e1 = halfedge[:, 0]

            w = np.array([[0, -1], [1, 0]], dtype=np.int_)
            v = (node[e1] - node[e0])@w
            val = np.sum(v*node[e0], axis=1)

            a = np.zeros(NC, dtype=self.ftype)
            np.add.at(a, halfedge[:, 1], val)
            a /=2
            return a
        elif self.ds.NV == 3:
            node = self.entity('node')
            cell = self.entity('cell')
            GD = self.geo_dimension()
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
            nv = np.cross(v1, v2)
            if GD == 2:
                a = nv/2.0
            elif GD == 3:
                a = np.sqrt(np.square(nv).sum(axis=1))/2.0
            return a

    def cell_barycenter(self, index=np.s_[:]): #TODO 三维
        """
        @brief 单元的重心。

        """
        GD = self.geo_dimension()
        NC = self.number_of_cells()

        node = self.entity('node') # DynamicArray
        halfedge = self.entity('halfedge') # DynamicArray

        cellflag = np.zeros(NC, dtype=np.bool_)
        cellflag[index] = True
        hflag = cellflag[halfedge[:, 1]]

        e0 = halfedge[halfedge[hflag, 3], 0]
        e1 = halfedge[hflag, 0]
        w  = np.array([[0, -1], [1, 0]], dtype=np.int_)
        v  = (node[e1] - node[e0])@w
        va = np.sum(v*node[e0], axis=1)
        ec = va.reshape(-1, 1)*(node[e1]+node[e0])/2

        a = np.zeros(NC, dtype=self.ftype)
        c = np.zeros((NC, GD), dtype=self.ftype)
        np.add.at(a, halfedge[hflag, 1], va)
        np.add.at(c, halfedge[hflag, 1], ec)
        a[index] /=2
        c[index] /=3*a[index, None]
        return c[index]

    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_normal(self, index=np.s_[:], node=None):
        """
        @brief Calculate the tangent vector with unit length of each edge.\
               See `Mesh.edge_tangent`.
        """
        v = self.edge_normal(index, node) 
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def interpolation_points(self, p: int, index=np.s_[:], scale=0.3):
        if self.ds.NV==3:
            cell = self.entity('cell')
            node = self.entity('node')
            if p == 1:
                return node

            if p > 1:
                NN = self.number_of_nodes()
                NE = self.number_of_edges()
                NC = self.number_of_cells()
                GD = self.geo_dimension()

                gdof = NN + NE*(p-1) + NC*(p-1)*(p-2)//2
                ipoints = np.zeros((gdof, GD), dtype=self.ftype)
                ipoints[:NN, :] = node


                edge = self.entity('edge')

                w = np.zeros((p-1, 2), dtype=np.float64)
                w[:, 0] = np.arange(p-1, 0, -1)/p
                w[:, 1] = w[-1::-1, 0]
                ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                        node[edge,:]).reshape(-1, GD)
            if p > 2:
                TD = self.top_dimension()
                multiIndex = self.multi_index_matrix(p, TD)
                isEdgeIPoints = (multiIndex == 0)
                isInCellIPoints = ~(isEdgeIPoints[:,0] | isEdgeIPoints[:,1] |
                        isEdgeIPoints[:,2])
                w = multiIndex[isInCellIPoints, :]/p

                ipoints[NN+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                        node[cell, :]).reshape(-1, GD)
            return ipoints # (gdof, GD)
        else:
            node = self.entity('node')

            if p == 1:
                return node

            gdof = self.number_of_global_ipoints(p)

            GD = self.geo_dimension()
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            start = 0
            ipoint = np.zeros((gdof, GD), dtype=self.ftype)
            ipoint[start:NN, :] = node

            start += NN

            edge = self.entity('edge')
            qf = self.integrator(p+1, etype='edge', qtype='lobatto')
            bcs = qf.quadpts[1:-1, :]
            ipoint[start:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
            start += (p-1)*NE

            if p == 2:
                ipoint[start:] = self.entity_barycenter('cell')
                return ipoint

            h = np.sqrt(self.cell_area())[:, None]*scale
            bc = self.entity_barycenter('cell')
            t = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]], dtype=self.ftype)
            t -= np.array([0.5, np.sqrt(3)/6.0], dtype=self.ftype)

            tri = np.zeros((NC, 3, GD), dtype=self.ftype)
            tri[:, 0, :] = bc + t[0]*h
            tri[:, 1, :] = bc + t[1]*h
            tri[:, 2, :] = bc + t[2]*h

            bcs = self.multi_index_matrix(p-2, 2)/(p-2)
            ipoint[start:] = np.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
            return ipoint


    def integrator(self, q, etype='cell', qtype='legendre'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            if qtype in {'legendre'}:
                from ..quadrature import GaussLegendreQuadrature
                return GaussLegendreQuadrature(q)
            elif qtype in {'lobatto'}:
                from ..quadrature import GaussLobattoQuadrature
                return GaussLobattoQuadrature(q)

    def integral(self, u, q=3, celltype=False):
        """
        @brief 多边形网格上的数值积分

        @param[in] u 被积函数, 需要两个参数 (x, index)
        @param[in] q 积分公式编号
        """
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        NC = self.number_of_cells()

        bcs, ws = self.integrator(q).get_quadrature_points_and_weights()

        bc = self.entity_barycenter('cell')
        tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]

        v1 = node[edge[:, 0]] - bc[edge2cell[:, 0]]
        v2 = node[edge[:, 1]] - bc[edge2cell[:, 0]]
        a = np.cross(v1, v2)/2.0

        pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = np.zeros(shape, dtype=np.float64)

        ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
        np.add.at(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            v1 = node[edge[isInEdge, 1]] - bc[edge2cell[isInEdge, 1]]
            v2 = node[edge[isInEdge, 0]] - bc[edge2cell[isInEdge, 1]]
            a = np.cross(v1, v2)/2.0

            pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
            np.add.at(e, edge2cell[isInEdge, 1], ee)

        if celltype is True:
            return e
        else:
            return e.sum(axis=0)

    def error(self, u, v, q=3, celltype=False, power=2):
        """
        @brief 在当前多边形网格上计算误差 \int |u - v|^power dx

        @param[in] u 函数
        @param[in] v 函数
        @param[in] q 积分公式编号
        """

        nu = len(inspect.signature(u).parameters)
        nv = len(inspect.signature(v).parameters)

        assert 1 <= nu <= 2
        assert 1 <= nv <= 2

        if (nu == 1) and (nv == 2):
            def efun(x, index):
                return np.abs(u(x) - v(x, index))**power
        elif (nu == 2) and (nv == 2):
            def efun(x, index):
                return np.abs(u(x, index) - v(x, index))**power
        elif (nu == 1) and (nv == 1):
            def efun(x, index):
                return np.abs(u(x) - v(x))**power
        else:
            def efun(x, index):
                return np.abs(u(x, index) - v(x))**power

        e = self.integral(efun, q, celltype=celltype)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)
        if celltype == False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e

    def delete_entity(self, isMarked, etype='node'):
        """
        @brief 删除 isMarked 对应的 etype 实体，调整 halfedge 中 etype 的编号
        """
        L = len(isMarked)
        l = L - isMarked.sum()
        idxmap = np.zeros(L, dtype=np.int_)
        idxmap[~isMarked] = np.arange(l)
        halfedge = self.entity('halfedge')
        if etype == 'node':
            halfedge[:, 0] = idxmap[halfedge[:, 0]]
        elif etype == 'cell':
            halfedge[:, 1] = idxmap[halfedge[: ,1]]
        elif etype == 'halfedge':
            halfedge[:, 2:] = idxmap[halfedge[:, 2:]]

    def mark_halfedge(self, isMarkedCell, method='poly'):
        clevel = self.celldata['level'] # 注意这里是所有的单元的层信息
        hlevel = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        if method == 'poly':
            while True:
                opp = halfedge[:, 4]
                isMarked = (~isMarkedCell[halfedge[:, 1]]) & isMarkedCell[halfedge[opp, 1]] 
                isMarked = isMarked & (clevel[halfedge[opp, 1]]>clevel[halfedge[:, 1]])
                isMarkedCell[halfedge[isMarked, 1]] = True
                if np.all(~isMarked):
                    break
            isMarkedHEdge = clevel[halfedge[:, 1]]>=clevel[halfedge[halfedge[:, 4], 1]]
            isMarkedHEdge = isMarkedHEdge & isMarkedCell[halfedge[:, 1]]
        elif method == 'quad':
            color = self.hedgecolor['color']
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2)|(color == 3)

            # 当前半边的层标记小于等于所属单元的层标记
            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0
            # 前一半边的层标记小于等于所属单元的层标记 
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1
            # 标记加密的半边的相对半边也需要标记 
            flag = ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]
            isMarkedHEdge[flag] = True
            while True:
                flag0 = isGreenHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 3]]
                flag1 = isRedHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 2]]
                flag2 = isOtherHEdge & ~isMarkedHEdge & (isMarkedHEdge[halfedge[:,
                    2]] | isMarkedHEdge[halfedge[:, 3]])
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'rg':
            color = self.halfedgedata['color']
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2)|(color == 3)

            # 当前半边的层标记小于等于所属单元的层标记
            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0
            # 前一半边的层标记小于等于所属单元的层标记 
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1 & (~isGreenHEdge)
            while True:
                flag = ~isMarkedCell[halfedge[:, 1]] & isMarkedHEdge & (~isRedHEdge)
                isMarkedCell[halfedge[flag, 1]] = True

                flag0 = isMarkedCell[halfedge[:, 1]] & (~isGreenHEdge) & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 2]] & isMarkedHEdge[
                        halfedge[:, 3]] & isRedHEdge & ~isMarkedHEdge
                flag2 = (isMarkedHEdge[halfedge[:, 2]] | isMarkedHEdge[
                        halfedge[:, 3]]) & isOtherHEdge & ~isMarkedHEdge
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'nvb':
            color = self.halfedgedata['color']
            isRedHEdge = color == 1
            isBlueHEdge = color == 0
            halfedge = self.entity('halfedge')

            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & isRedHEdge 
            while True:
                flag0 = isRedHEdge & (isMarkedHEdge[halfedge[:,
                    2]]|isMarkedHEdge[halfedge[:, 3]]) & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        return isMarkedHEdge

    def refine_halfedge(self, isMarkedHEdge, newnode=None):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NHE = self.ds.number_of_halfedges()

        hlevel = self.halfedgedata['level']
        halfedge = self.ds.halfedge
        node = self.node
        hedge = self.ds.hedge
        isMainHEdge = self.ds.main_halfedge_flag()

        # 添加点
        flag = isMarkedHEdge & isMainHEdge # 即是主半边, 也是标记加密的半边
        pre = halfedge[flag, 3]
        NE1 = flag.sum()
        newNode = node.increase_size(NE1)
        if newnode is None:
            newNode[:] = (node[halfedge[flag, 0]] + node[halfedge[pre, 0]])/2
        elif isinstance(newnode, np.ndarray):
            newNode[:] = newnode

        edge2NewNode = np.zeros(NHE, dtype=np.int_)
        edge2NewNode[flag] = np.arange(NE1)+NN
        edge2NewNode[halfedge[flag, 4]] = np.arange(NE1)+NN

        #细分边
        NHE1 = isMarkedHEdge.sum()
        current, = np.where(isMarkedHEdge)
        opp = halfedge[current, 4]
        pre = halfedge[current, 3]
        flag = current!=opp # 非边界半边

        halfedge[pre, 2] = np.arange(NHE, NHE+NHE1)
        halfedge[current, 3] = np.arange(NHE, NHE+NHE1)
        halfedge[current[flag], 4] = halfedge[opp[flag], 3]

        newHalfedge = halfedge.increase_size(NHE1)
        newHalfedge[:, 0] = edge2NewNode[current] 
        newHalfedge[:, 1] = halfedge[:NHE][isMarkedHEdge, 1]
        newHalfedge[:, 2] = current
        newHalfedge[:, 3] = pre
        newHalfedge[:, 4] = np.arange(NHE, NHE+NHE1)
        newHalfedge[flag, 4] = opp[flag]

        #修改半边层
        hlevel[isMarkedHEdge] +=1
        hlevel.extend(np.zeros(NHE1, dtype=np.int_))

        self.ds.reinit()
        return NHE1

    def coarsen_halfedge(self, isMarkedHEdge):
        """
        @brief 将标记的半边删除
        """
        self.ds.reinit()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        node     = self.entity('node')
        halfedge = self.entity('halfedge')
        hlevel   = self.halfedgedata['level']
        isMainHEdge = self.ds.main_halfedge_flag()

        nex = halfedge[isMarkedHEdge, 2]
        pre = halfedge[isMarkedHEdge, 3]
        opp = halfedge[isMarkedHEdge, 4]
        flag = nex!=halfedge[nex, 4]

        halfedge[pre, 2] = nex
        halfedge[nex, 3] = pre
        halfedge[nex[flag], 4] = opp[flag]

        isRNode = np.zeros(NN, dtype=np.bool_)
        isRNode[halfedge[isMarkedHEdge, 0]] = True

        #更新节点
        node.adjust_size(isRNode)
        self.delete_entity(isRNode, etype='node')

        # 更新halfedge
        self.delete_entity(isMarkedHEdge, etype='halfedge')
        halfedge.adjust_size(isMarkedHEdge)

        # 更新半边层
        hlevel[nex]-=1
        hlevel.adjust_size(isMarkedHEdge)

        self.ds.reinit()

    def _refine_poly_cell_(self, isMarkedCell, isStartHEdge, options={}):
        NN  = self.number_of_nodes()
        NC  = self.ds.number_of_cells()
        NC1 = isStartHEdge.sum() 
        NN1 = isMarkedCell.sum()
        NHE = self.ds.number_of_halfedges()

        clevel   = self.celldata['level']
        hlevel   = self.halfedgedata['level']
        halfedge = self.entity('halfedge')

        #生成新的节点
        self.node.extend(self.cell_barycenter(index=isMarkedCell))

        cell2newNode = np.zeros(NC, dtype=np.int_)
        cell2newNode[isMarkedCell[:NC]] = np.arange(NN, NN+NN1)

        #生成新的单元
        current = np.where(isStartHEdge)[0]
        pre  = halfedge[current, 3]
        nex  = halfedge[current, 2]
        cidx = halfedge[current, 1]
        halfedge[current, 1] = np.arange(NC, NC+NC1)
        halfedge[nex, 1] = np.arange(NC, NC+NC1)
        isNotOK = np.ones_like(nex, dtype=np.bool_)
        while np.any(isNotOK):
            isNotOK = ~isStartHEdge[halfedge[nex, 2]]
            nex[isNotOK] = halfedge[nex[isNotOK], 2]
            halfedge[nex, 1] = np.arange(NC, NC+NC1)

        ppre = halfedge[halfedge[:, 3], 3]
        ppre[halfedge[nex, 2]] = current
        ppre = ppre[current]
        halfedge[current, 3] = np.arange(NHE+NC1, NHE+NC1*2)
        halfedge[pre, 2] = np.arange(NHE, NHE+NC1)

        halfedgeNew = halfedge.increase_size(NC1*2)
        halfedgeNew[:NC1, 0] = cell2newNode[cidx]
        halfedgeNew[:NC1, 1] = halfedge[pre, 1]
        halfedgeNew[:NC1, 2] = halfedge[ppre, 3]
        halfedgeNew[:NC1, 3] = pre
        halfedgeNew[:NC1, 4] = halfedge[current, 3]

        halfedgeNew[NC1:, 0] = halfedge[pre, 0]
        halfedgeNew[NC1:, 1] = halfedge[current, 1]
        halfedgeNew[NC1:, 2] = current
        halfedgeNew[NC1:, 3] = halfedge[nex, 2]
        halfedgeNew[NC1:, 4] = halfedge[pre, 2]

        #修改单元编号
        idx     = np.unique(halfedge[:, 1])
        cidxmap = np.arange(NC+NC1)
        cidxmap[idx]   = np.arange(idx.size)
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        #单元层
        clevel1 = clevel[cidx]
        clevelNew = clevel.adjust_size(isMarkedCell, int(NC1))
        clevelNew[:] = clevel1+1

        #半边层
        hlevel.extend(np.zeros(NC1*2, dtype=np.int_))

        self.ds.reinit()

        if 'numrefine' in options:
            num = options['numrefine']
            num = np.r_[num, np.zeros(NC1, dtype=np.int_)]
            num[-NC1:] = num[cidx]-1#current所属的单元加密次数-1

            #若不需要加密的单元加密了, 将他的子单元加密次数设为0
            flag = np.zeros(len(num), dtype=np.bool_)
            flag[-NC1:] = True
            flag = flag & (num<0)
            num[flag] = 0

            options['numrefine'] = num[idx]

    def _coarsen_poly_cell(self, isMarkedCell, isRNode, options={}):
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hcell = self.ds.hcell
        hedge = self.ds.hedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        isMainHEdge = self.ds.main_halfedge_flag()

        flag = isMarkedCell[halfedge[:, 1]]
        np.logical_and.at(isRNode, halfedge[:, 0], flag)
        nn = isRNode.sum()
        newNode = node[isRNode]

        # 重新标记要移除的单元
        isMarkedHEdge = isRNode[halfedge[:, 0]]
        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
        isMarkedCell = np.zeros(NC+nn, dtype=np.bool_)
        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        nc = np.sum(~isMarkedCell[:NC])# 没有被标记的单元个数

        # 更新单元层
        ncl = np.zeros(NN, dtype=self.itype)
        ncl[halfedge[:, 0]] = clevel[halfedge[:, 1]]-1
        clevel.adjust_size(isMarkedCell[:NC], ncl[isRNode])

        # 更新单元编号
        cidxmap = np.arange(NC)
        nidxmap = np.arange(NN)
        nidxmap[isRNode] = range(nc, nc+nn)
        isRHEdge = isRNode[halfedge[:, 0]]
        cidxmap[halfedge[isRHEdge, 1]] = nidxmap[halfedge[isRHEdge, 0]] #修改标记单元的编号
        cidxmap[~isMarkedCell[:NC]] = np.arange(nc) #修改未标记单元的编号
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        if 'numrefine' in options:
            num0 = np.zeros(nc+nn)-10000
            num = options['numrefine']
            num[isMarkedCell[:NC]]+=1
            np.maximum.at(num0, cidxmap, num)
            options['numrefine'] = num0

        # 修改保留半边的下一个边, 上一个边
        nex = halfedge[:, 2]
        pre = halfedge[:, 3]
        opp = halfedge[:, 4]
        flag1 = isMarkedHEdge[nex]
        halfedge[flag1, 2] = nex[opp[nex[flag1]]]
        flag2 = isMarkedHEdge[pre]
        halfedge[flag2, 3] = pre[opp[pre[flag2]]]

        # 对半边重新编号
        self.delete_entity(isMarkedHEdge, etype='halfedge')
        #更新节点编号
        self.delete_entity(isRNode, etype='node')
        # 更新halfedge
        halfedge.adjust_size(isMarkedHEdge)
        # 更新半边层
        hlevel.adjust_size(isMarkedHEdge)
        #更新节点
        node.adjust_size(isRNode)

        self.ds.reinit()
        return isMarkedHEdge, isRNode, newNode

    def refine_poly(self, isMarkedCell=None, options={'disp': True}):
        clevel   = self.celldata['level']
        hlevel   = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        isMainHEdge = self.ds.main_halfedge_flag()
        NC = self.number_of_cells()

        isMarkedCell = np.ones(NC, dtype=np.bool_) if isMarkedCell is None else isMarkedCell
        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='poly')

        opp = halfedge[:, 4]
        cidx = halfedge[:, 1]
        flag = (clevel[cidx]+hlevel[:]+1==clevel[cidx[opp]]) & isMarkedCell[cidx]
        mark0 = np.where(flag)[0]
        mark1 = np.where(isMarkedHEdge)[0]

        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
        self.refine_halfedge(isMarkedHEdge)

        # isStartHedge : 引导生成加密后的单元
        NHE = self.ds.number_of_halfedges()
        isStartHEdge = np.zeros(NHE, dtype=np.bool_)
        isStartHEdge[mark1] = True
        isStartHEdge[halfedge[mark0, 2]] = True

        self._refine_poly_cell_(isMarkedCell, isStartHEdge, options=options)

    def coarsen_poly(self, isMarkedCell, i=0, options={'disp': True}):

        NC = self.number_of_cells()
        NN = self.number_of_nodes().copy()
        NE = self.number_of_edges()
        NHE = self.ds.number_of_halfedges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        isMainHEdge = self.ds.main_halfedge_flag()

        # 保证相邻单元层数小于等于1
        while True:
            opp = halfedge[:, 4]
            isRNode = np.ones(NN, dtype=np.bool_)
            flag = (hlevel == hlevel[opp])&(clevel[halfedge[:, 1]]>0)&(opp!=np.arange(NHE))
            flag = flag & isMarkedCell[halfedge[:, 1]]
            np.logical_and.at(isRNode, halfedge[:, 0], flag)
            flag = isRNode[halfedge[:, 0]]
            isMarkedCell[:] = False
            isMarkedCell[halfedge[flag, 1]] = True

            isMarked = (isMarkedCell[halfedge[:, 1]]) & (~isMarkedCell[halfedge[opp, 1]])
            isMarked = isMarked & (clevel[halfedge[opp, 1]]>clevel[halfedge[:, 1]])
            isMarkedCell[halfedge[isMarked, 1]] = False
            if np.all(~isMarked):
                break

        # 可以移除的网格节点
        isRNode = np.ones(NN, dtype=np.bool_)
        flag = (hlevel == hlevel[halfedge[:, 4]])&(clevel[halfedge[:, 1]]>0)&(halfedge[:, 4]!=np.arange(NHE))
        np.logical_and.at(isRNode, halfedge[:, 0], flag)

        #粗化单元
        self._coarsen_poly_cell(isMarkedCell, isRNode, options=options)

        NHE = self.ds.number_of_halfedges()
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        flag = (opp[nex[opp[nex]]] == np.arange(NHE))|(halfedge[:, 4]==np.arange(NHE))
        flag = flag & (hlevel[:]==0) & (hlevel[halfedge[:, 2]]>0)
        self.coarsen_halfedge(flag)

    def _refine_tri_cell_(self, isMarkedCell, isMarkedHEdge, options={}):
        NC  = self.number_of_cells()
        NC1 = isMarkedHEdge.sum()
        NHE = self.ds.number_of_halfedges()

        halfedge = self.entity("halfedge")
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        current, = np.where(isMarkedHEdge)
        pre = halfedge[current, 3]
        ppre = halfedge[pre, 3]
        nex = halfedge[current, 2]
        nnex = halfedge[nex, 2]

        cidx = halfedge[current, 1]
        halfedge[nnex, 3] = np.arange(NHE+NC1, NHE+NC1*2)
        halfedge[pre, 2] = np.arange(NHE+NC1, NHE+NC1*2)
        halfedge[current, 1] = np.arange(NC, NC+NC1)
        halfedge[current, 3] = np.arange(NHE, NHE+NC1)
        halfedge[nex, 1] = np.arange(NC, NC+NC1)
        halfedge[nex, 2] = np.arange(NHE, NHE+NC1)

        halfedgeNew = halfedge.increase_size(NC1*2)
        halfedgeNew[:NC1, 0] = halfedge[pre, 0]
        halfedgeNew[:NC1, 1] = np.arange(NC, NC+NC1)
        halfedgeNew[:NC1, 2] = current
        halfedgeNew[:NC1, 3] = nex
        halfedgeNew[:NC1, 4] = np.arange(NHE+NC1, NHE+NC1*2)

        halfedgeNew[NC1:, 0] = halfedge[nex, 0]
        halfedgeNew[NC1:, 1] = cidx
        halfedgeNew[NC1:, 2] = nnex
        halfedgeNew[NC1:, 3] = pre
        halfedgeNew[NC1:, 4] = np.arange(NHE, NHE+NC1)

        if ('HB' in options) and (options['HB'] is not None):
            NC0 = NC-cstart
            HB0 = np.zeros([NC0+NC1, 2], dtype=np.int_)
            HB0[:, 0] = np.arange(NC0+NC1)
            HB0[:NC0, 1] = np.arange(NC0)
            HB0[NC0:, 1] = cidx-cstart
            HB = options['HB']
            HB0[:, 1] = HB[HB0[:, 1], 1]
            options['HB'] = HB0

        flag = isMarkedHEdge[halfedgeNew[NC1:, 2]]
        current = np.where(flag)[0]+NC1
        nexpre = halfedge[halfedgeNew[current, 2], 3]
        prenex = halfedge[halfedgeNew[current, 3], 2]
        halfedgeNew[current, 2] = halfedge[nexpre, 4]
        halfedgeNew[current, 3] = halfedge[prenex, 4]

        if ('numrefine' in options) & (NC1>0):
            num = options['numrefine']
            num = np.r_[num, np.zeros(NC1)]

            #若加密次数小于等于0, 却被加密了, 子单元加密次数设为 0.
            num[cidx]-=1
            flag = np.zeros(len(num), dtype=np.bool_)
            flag[cidx] = True
            flag  = flag & (num<0)
            num[flag]=0

            num[-NC1:] = num[cidx]
            options['numrefine'] = num

        #单元层
        clevelNew = clevel.increase_size(NC1)
        clevel[cidx] +=1
        clevelNew[:] = clevel[cidx]

        #半边层
        hlevel.extend(np.zeros(NC1*2, dtype=np.int_))

        self.ds.reinit()

    def _coarsen_triangle_cell_(self, isMarkedCell, isMarkedHEdge, options={}):
        NC  = self.number_of_cells()
        NHE = self.ds.number_of_halfedges()

        halfedge = self.entity("halfedge")
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        flag = (halfedge[:, 1]<halfedge[halfedge[:, 4], 1]) & isMarkedHEdge
        isRCell = np.zeros(NC, dtype=np.bool_)
        isRCell[halfedge[flag, 1]] = True

        #修改下一个边和上一个边
        for i in range(2):
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            opp = halfedge[:, 4]
            flag1 = isMarkedHEdge[nex]
            halfedge[flag1, 2] = nex[opp[nex[flag1]]]
            flag2 = isMarkedHEdge[pre]
            halfedge[flag2, 3] = pre[opp[pre[flag2]]]

        #修改单元
        flag = isRCell[halfedge[: ,1]] & isMarkedHEdge

        nc = isRCell.sum()
        opp = halfedge[:, 4]
        cidxmap = np.arange(NC)
        cidxmap[halfedge[opp[flag], 1]] = cidxmap[halfedge[flag, 1]]
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        cell0 = np.unique(halfedge[:, 1])
        cidxmap0 = np.arange(NC)
        cidxmap0[cell0] = np.arange(len(cell0))
        halfedge[:, 1] = cidxmap0[halfedge[:, 1]]

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC-cstart, 2), dtype=np.int)
            HB[:, 0] = np.arange(NC-cstart)
            HB[:, 1] = cidxmap0[cidxmap[cstart:]]-cstart
            options['HB'] = HB

        if 'numrefine' in options:
            num = options['numrefine']
            np.maximum.at(num, cidxmap, num)
            num[halfedge[flag, 1]] += 1
            options['numrefine'] = num[cell0]

        #更新单元层
        flag = cidxmap!=np.arange(NC)
        clevel[isRCell]-=1
        clevel.adjust_size(flag)

        #更新半边层
        hlevel.adjust_size(isMarkedHEdge)

        # 对半边重新编号
        self.delete_entity(isMarkedHEdge, etype='halfedge')

        # 更新halfedge
        halfedge.adjust_size(isMarkedHEdge)

        self.ds.reinit()
        return isMarkedHEdge

    def refine_triangle_rg(self, isMarkedCell=None, options={}):
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NHE = self.ds.number_of_halfedges()

        color = self.halfedgedata.get('color')
        node = self.entity('node')
        halfedge = self.entity("halfedge")
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        hedge = self.ds.hedge
        hcell = self.ds.hcell

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if color is None:
            color = np.zeros(NHE, dtype=np.int_)
            self.halfedgedata['color'] = color


        isBlueHEdge = color == 3
        isYellowHedge = color == 2

        #得到加密半边并加密
        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell, method='rg')
        isMarkedHEdge = isMarkedHEdge0 & ((color==0)|(color==1))

        NHE1 = self.refine_halfedge(isMarkedHEdge) # 加密半边得到的新半边的个数
        isMainHEdge = self.ds.main_halfedge_flag()

        color = np.r_[color, np.zeros(NHE1, dtype = np.int_)]

        #标记的蓝色单元变成红色单元
        flag = isBlueHEdge & isMarkedHEdge0
        NE1 = flag.sum()

        NHE = self.ds.number_of_halfedges()
        ################## 绿色单元转变红色单元 ##################
        halfedgeNew = halfedge.increase_size(NE1*4)
        isMainHEdgeNew = np.zeros(NE1*4, dtype=np.bool_)

        current, = np.where(flag)
        pre = halfedge[current, 3]
        ppre = halfedge[pre, 3]
        pppre = halfedge[ppre, 3]
        opp = halfedge[current, 4]

        halfedge[current, 3] = ppre#修改蓝色半边位置
        halfedge[current, 4] = np.arange(NHE, NHE+NE1)
        halfedge[ppre, 2] = current
        halfedge[pre, 1] = np.arange(NC, NC+NE1)
        halfedge[pre, 2] = halfedge[opp, 2]
        halfedge[pre, 3] = np.arange(NHE+NE1*2, NHE+NE1*3)
        halfedgeNew[:NE1, 0] = halfedge[ppre, 0]
        halfedgeNew[:NE1, 1] = np.arange(NC+NE1, NC+NE1*2)
        halfedgeNew[:NE1, 2] = np.arange(NHE+NE1*3, NHE+NE1*4)
        halfedgeNew[:NE1, 3] = np.arange(NHE+NE1, NHE+NE1*2)
        halfedgeNew[:NE1, 4] = current
        halfedgeNew[NE1*2:NE1*3, 0] = halfedge[ppre, 0]
        halfedgeNew[NE1*2:NE1*3, 1] = np.arange(NC, NC+NE1)
        halfedgeNew[NE1*2:NE1*3, 2] = pre
        halfedgeNew[NE1*2:NE1*3, 3] = halfedge[opp, 2]
        halfedgeNew[NE1*2:NE1*3, 4] = np.arange(NHE+NE1*3, NHE+NE1*4)
        isMainHEdgeNew[:NE1] = ~isMainHEdge[current]
        isMarkedHEdge[pre] = False

        current = opp.copy()
        nex = halfedge[current, 2]
        nnex = halfedge[nex, 2]
        nnnex = halfedge[nnex, 2]
        opp = halfedge[current, 4]

        halfedge[current, 0] = halfedge[nex, 0]#修改黄色半边位置
        halfedge[current, 2] = nnex
        halfedge[current, 4] = np.arange(NHE+NE1, NHE+NE1*2)
        halfedge[nnex, 3] = current
        halfedge[nex, 1] = np.arange(NC, NC+NE1)
        halfedge[nex, 2] = np.arange(NHE+NE1*2, NHE+NE1*3)
        halfedge[nex, 3] = pre
        halfedgeNew[NE1:NE1*2, 0] = halfedge[opp, 0]
        halfedgeNew[NE1:NE1*2, 1] = np.arange(NC+NE1, NC+NE1*2)
        halfedgeNew[NE1:NE1*2, 2] = np.arange(NHE, NHE+NE1)
        halfedgeNew[NE1:NE1*2, 3] = np.arange(NHE+NE1*3, NHE+NE1*4)
        halfedgeNew[NE1:NE1*2, 4] = current
        halfedgeNew[NE1*3:NE1*4, 0] = halfedge[nex, 0]
        halfedgeNew[NE1*3:NE1*4, 1] = np.arange(NC+NE1, NC+NE1*2)
        halfedgeNew[NE1*3:NE1*4, 2] = np.arange(NHE+NE1, NHE+NE1*2)
        halfedgeNew[NE1*3:NE1*4, 3] = np.arange(NHE, NHE+NE1)
        halfedgeNew[NE1*3:NE1*4, 4] = np.arange(NHE+NE1*2, NHE+NE1*3)
        isMainHEdgeNew[NE1:NE1*2] = ~isMainHEdge[current]
        isMainHEdgeNew[NE1*3:NE1*4] = True
        isMarkedHEdge[nnex] = False

        #中心单元编号最小
        cidxmap = np.arange(NC+NE1*2)
        cidxmap[halfedge[current, 1]] = np.arange(NC+NE1, NC+NE1*2)
        cidxmap[NC+NE1:NC+NE1*2] = halfedge[current, 1]
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        #半边层
        hlevel.extend(np.zeros(NE1*4, dtype=np.int_))

        #单元层
        clevel.extend(clevel[halfedge[opp, 1]])
        clevel.extend(clevel[halfedge[opp, 1]])

        #修改半边颜色
        idx = np.r_[current, opp, nex, pre, ppre, nnex, pppre, nnnex]
        color = np.r_[color, np.zeros(NE1*4, dtype=np.int_)]
        color[idx] = 0

        self.ds.reinit()
        ##########################################################

        #标记要生成新单元的单元
        NV = self.ds.number_of_vertices_of_cells()
        isBlueCell = NV == 4
        isNewCell = (NV == 4)|(NV == 6)

        NC+=NE1*2
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NHE1+NE1*4, dtype=np.bool_)]

        #修改半边颜色(1)
        flag = isMarkedHEdge & isBlueCell[halfedge[:, 1]]
        tmp = np.where(flag)[0]
        #生成新单元
        isMarkedHEdge = isMarkedHEdge & isNewCell[halfedge[:, 1]]#既是标记边又对应标记单元
        NC1 = isMarkedHEdge.sum()
        self._refine_tri_cell_(isNewCell, isMarkedHEdge, options=options)

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[color==3, 4]] = 2
        color[halfedge[color==2, 3]] = 1
        self.halfedgedata['color'] = color

    def coarsen_triangle_rg(self, isMarkedCell, options={}):

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.halfedgedata['color']
        halfedge = self.entity('halfedge')

        #标记红色单元要删除的半边
        flag = (clevel[halfedge[:, 1]]==clevel[halfedge[halfedge[:, 4], 1]])
        flag = (hlevel[:]==0) & (clevel[halfedge[:, 1]] >0) & flag & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]#自身和对边都是标记单元的半边

        isRCell = np.ones(NC, dtype=np.bool_)
        np.logical_and.at(isRCell, halfedge[:, 1], flag)#周围都被标记的中心单元

        isMarkedHEdge = isRCell[halfedge[:, 1]]
        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True

        #标记绿色单元要删除的半边
        flag = ((color==2) | (color==3)) & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]
        isMarkedHEdge[flag] = True

        #删除单元
        isMarkedHEdge = self._coarsen_triangle_cell_(isMarkedCell, isMarkedHEdge, options=options)

        color = color[~isMarkedHEdge]#更新颜色
        color[color==1] = 0
        color[halfedge[color==3, 2]] = 1
        color[halfedge[color==2, 3]] = 1

        #进一步标记需要删除的边
        isBDHEdge = self.ds.boundary_halfedge_flag()
        NC = self.number_of_cells()
        flag = (halfedge[halfedge[halfedge[halfedge[:, 2], 4], 2],
            4]==np.arange(len(halfedge))) & (hlevel[halfedge[:, 4]]!=0)
        flag = flag |(isBDHEdge & isBDHEdge[halfedge[:, 2]])
        flag = flag & (hlevel[:]==0) & (hlevel[halfedge[:, 2]] > 0)

        while True:
            count = np.zeros(NC, dtype=np.int_)
            np.add.at(count, halfedge[:, 1], flag)

            NV = self.ds.number_of_vertices_of_cells()
            flag0 = (count==1) & (NV==6)
            flag[flag0[halfedge[:, 1]]] = False
            flag[flag0[halfedge[halfedge[:, 4], 1]]] = False
            if (~flag0).all():
                break

        #修改颜色
        color[halfedge[flag, 4]] = 0
        color = color[~flag]
        self.coarsen_halfedge(flag)

        #标记要生成新单元的单元
        NV = self.ds.number_of_vertices_of_cells()
        isBlueCell = NV==4
        isRedCell = NV==6
        isNewCell = (NV == 4)|(NV == 6)

        #生成新单元
        isBDHEdge = self.ds.boundary_halfedge_flag()
        flag0 = (hlevel[:]>0) & (hlevel[halfedge[:, 3]]==0) 
        flag1 = flag0 & (hlevel[:]!=hlevel[halfedge[:, 4]]) & (hlevel[halfedge[halfedge[:, 3], 4]]>0)
        flag1 = (flag1 & (~isBDHEdge)) | (flag0 & isBDHEdge & isRedCell[halfedge[:, 1]])
        flag = flag1 & isNewCell[halfedge[:, 1]]#既是标记边又对应标记单元
        tmp = flag1 & isBlueCell[halfedge[:, 1]]#既是标记边又对应标记单元
        tmp, = np.where(tmp)
        NC1 = flag.sum()

        flag = (hlevel[:]==0) & (hlevel[halfedge[:, 2]]>0) & (hlevel[halfedge[:,
            4]]>0) & isBlueCell[halfedge[:, 1]]
        flag = flag | ((hlevel[:]==0) & isRedCell[halfedge[:, 1]])
        flag = flag[halfedge[:, 3]]

        self._refine_tri_cell_(isNewCell, flag, options=options)

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[color==3, 4]] = 2
        color[halfedge[color==2, 3]] = 1
        self.halfedgedata['color'] = color

    def refine_triangle_nvb(self, isMarkedCell=None, options={}):
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NHE = self.ds.number_of_halfedges()

        color = self.halfedgedata.get('color')
        node = self.entity('node')
        halfedge = self.entity('halfedge')

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if color is None:
            color = np.zeros(NHE, dtype=np.int_)
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            l = node[halfedge[:, 0]]-node[halfedge[pre, 0]]
            l = np.linalg.norm(l, axis=1)
            color[(l>l[nex]) & (l>l[pre])] = 1
            self.halfedgedata['color'] = color

        hedge = self.ds.hedge
        hcell = self.ds.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC, 2), dtype=np.int_)
            HB[:, 0] = np.arange(NC)
            HB[:, 1] = np.arange(NC)
            options['HB'] = HB

        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='nvb')
        flag = np.array(True)
        while flag.any():
            #加密半边
            flag = isMarkedHEdge & (color==1)
            flag = flag & flag[halfedge[:, 4]]

            #加密半边
            NE1 = self.refine_halfedge(flag)

            #加密单元
            NV = self.ds.number_of_vertices_of_cells()
            isNewCell = NV==4

            flag = np.r_[flag, np.zeros(NE1, dtype=np.bool_)]
            #flag = flag & (halfedge[:, 1]>=cstart)

            NE2 = flag.sum()
            isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NE1+NE2*2, dtype=np.bool_)]

            #修改半边颜色
            color = np.r_[color, np.zeros(NE1, dtype=np.int_)]
            color[flag]=0
            color[halfedge[flag, 2]] = 1
            color[halfedge[halfedge[flag, 3], 3]] = 1
            color = np.r_[color, np.zeros(NE2*2, dtype=np.int_)]
            self.halfedgedata['color'] = color

            self._refine_tri_cell_(isNewCell, flag, options=options)


    def coarsen_triangle_nvb(self, isMarkedCell, options={}):
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NHE = self.ds.number_of_halfedges()
        hcell = self.ds.hcell
        hedge = self.ds.hedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.halfedgedata['color']
        node = self.entity('node')
        halfedge = self.entity('halfedge')
        isMainHEdge = self.ds.main_halfedge_flag()

        nex, pre, opp = halfedge[:, 2], halfedge[:, 3], halfedge[:, 4]

        #标记要删除的半边
        isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & (hlevel[:]==0) & (color[:]==0)
        isMarkedHEdge = isMarkedHEdge & (clevel[halfedge[:, 1]]>0)
        isMarkedHEdge = isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]

        #只考虑对着的四个四边形都被标记的情况或者两个边界上的三角形都被标记的情况
        isMarkedHEdge = isMarkedHEdge & (color[halfedge[:, 3]] == 1)

        isMarkedHEdge = isMarkedHEdge & (isMarkedHEdge[opp[nex[opp[nex]]]] | (opp[nex] == nex))
        isMarkedHEdge[opp[isMarkedHEdge]] = True

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC-cstart, 2), dtype=np.int)
            HB[:, 0] = np.arange(NC-cstart)
            HB[:, 1] = np.arange(NC-cstart)
            options['HB'] = HB

        isMarkedHEdge = self._coarsen_triangle_cell_(isMarkedCell, isMarkedHEdge, options=options)

        #修改颜色
        color = color[~isMarkedHEdge]
        flag = (color==1) & (color==1)[halfedge[:, 2]]
        color[flag] = 0
        color[halfedge[flag, 2]] = 0
        color[halfedge[flag, 3]] = 1

        #进一步标记要移除的半边
        NV = self.ds.number_of_vertices_of_cells()
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        isBDHEdge = self.ds.boundary_halfedge_flag()
        flag = (opp[nex[opp[nex]]] == np.arange(len(halfedge))) |(isBDHEdge & isBDHEdge[nex])
        flag = flag & (color==1)[halfedge[:, 2]] & (NV[halfedge[:, 1]]==4)
        #flag[opp[nex[flag]]]=True

        #粗化半边
        self.coarsen_halfedge(flag)

        #修改颜色
        color = color[~flag]
        self.halfedgedata['color'] = color

    def refine_cell(self, isMarked, isMarkedHEdge, method='quad',
            bc=None, options={}):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level']
        hedge = self.ds.hedge
        hcell = self.ds.hcell

        NC1 = isMarkedHEdge.sum()
        NN1 = isMarked.sum()
        isMainHEdge = self.ds.main_halfedge_flag()

        if method=='quad':
            isMarkedCell = isMarked
            #生成新的节点
            if bc is None:
                bc = self.cell_barycenter()
                node.extend(bc[isMarkedCell[cstart:NC]])
            else:
                node.extend(bc[isMarkedCell[:NC]])
            cell2newNode = np.zeros(NC, dtype=np.int_)
            cell2newNode[isMarkedCell[:NC]] = np.arange(node.size)[-NN1:]

            #生成新的单元
            current, = np.where(isMarkedHEdge)
            pre = halfedge[current, 3]
            nex = halfedge[current, 2]

            cidx = halfedge[current, 1]
            halfedge[current, 1] = np.arange(NC, NC+NC1)
            halfedge[nex, 1] = np.arange(NC, NC+NC1)
            isNotOK = ~isMarkedHEdge[halfedge[nex, 2]]
            idx = nex.copy()
            while isNotOK.any():
                idx[isNotOK] = halfedge[nex[isNotOK], 2]
                flag = isMarkedHEdge[idx]
                nex[~flag] = idx[~flag]
                halfedge[nex, 1] = np.arange(NC, NC+NC1)
                isNotOK[flag]=False
            ppre = halfedge[halfedge[:, 3], 3]
            ppre[halfedge[nex, 2]] = current
            ppre = ppre[current]
            halfedge[current, 3] = np.arange(NE*2+NC1, NE*2+NC1*2)
            halfedge[pre, 2] = np.arange(NE*2, NE*2+NC1)

            halfedgeNew = halfedge.increase_size(NC1*2)
            halfedgeNew[:NC1, 0] = cell2newNode[cidx]
            halfedgeNew[:NC1, 1] = halfedge[pre, 1]
            halfedgeNew[:NC1, 2] = halfedge[ppre, 3]
            halfedgeNew[:NC1, 3] = pre
            halfedgeNew[:NC1, 4] = halfedge[current, 3]

            halfedgeNew[NC1:, 0] = halfedge[pre, 0]
            halfedgeNew[NC1:, 1] = halfedge[current, 1]
            halfedgeNew[NC1:, 2] = current
            halfedgeNew[NC1:, 3] = halfedge[nex, 2]
            halfedgeNew[NC1:, 4] = halfedge[pre, 2]

            #修改单元编号
            cidxmap = np.arange(NC+NC1)
            idx = np.unique(halfedge[:, 1])
            NC2 = idx.shape[0]
            cidxmap[idx] = np.arange(NC2)
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            if 'numrefine' in options:
                num = options['numrefine']
                num = np.r_[num, np.zeros(NC1, dtype=np.int_)]
                num[-NC1:] = num[cidx]-1#current所属的单元加密次数-1

                #若不需要加密的单元加密了, 将他的子单元加密次数设为0
                flag = np.zeros(len(num), dtype=np.bool_)
                flag[-NC1:] = True
                flag = flag & (num<0)
                num[flag] = 0

                options['numrefine'] = num[idx]

            if ('HB' in options) and (options['HB'] is not None):
                isNonMarkedCell = ~isMarkedCell
                flag0 = isNonMarkedCell[cstart:]
                flag1 = isMarkedCell[cstart:]
                NHB0 = flag0.sum()
                NHE = len(halfedge)
                NHB = NHB0 + NHE
                HB = np.zeros((NHB, 2), dtype=np.int)
                HB[:, 0] = range(NHB)
                HB[0:NHB0, 1] = np.arange(len(flag0))[flag0]
                HB[NHB0:,  1] = cellidx - cellstart
                HB0 = HB.copy()
                HB0[:, 1] = options['HB'][HB0[:,1], 1]
                options['HB'] = HB0

            #增加主半边
            hedge.extend(np.arange(NE*2, NE*2+NC1))

            #更新subdomain
            subdomainNew = subdomain.adjust_size(isMarkedCell, int(NC1))
            subdomainNew[:] = subdomain[cidx]

            #更新起始边
            hcell.increase_size(NC2-NC)
            hcell[halfedge[:, 1]] = np.arange(len(halfedge)) # 的编号

            #单元层
            clevel1 = clevel[cidx]
            clevelNew = clevel.adjust_size(isMarkedCell, int(NC1))
            clevelNew[:] = clevel1+1

            #半边层
            hlevel.extend(np.zeros(NC1*2, dtype=np.int_))

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2

    def coarsen_cell(self, isMarkedCell, isMarked, method='quad', options={}):

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hcell = self.ds.hcell
        hedge = self.ds.hedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        if method=='quad':
            isRNode = isMarked
            flag = isMarkedCell[halfedge[:, 1]]
            np.logical_and.at(isRNode, halfedge[:, 0], flag)
            nn = isRNode.sum()
            newNode = node[isRNode]

            # 重新标记要移除的单元
            isMarkedHEdge = isRNode[halfedge[:, 0]]
            isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
            isMarkedCell = np.zeros(NC+nn, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            nc = np.sum(~isMarkedCell[:NC])# 没有被标记的单元个数

            # 更新subdomain
            nsd = np.zeros(NN, dtype=self.itype)
            nsd[halfedge[:, 0]] = subdomain[halfedge[:, 1]]
            subdomain.adjust_size(isMarkedCell[:NC], nsd[isRNode])

            # 更新单元层
            ncl = np.zeros(NN, dtype=self.itype)
            ncl[halfedge[:, 0]] = clevel[halfedge[:, 1]]-1
            clevel.adjust_size(isMarkedCell[:NC], ncl[isRNode])

            # 更新单元编号
            cidxmap = np.arange(NC)
            nidxmap = np.arange(NN)
            nidxmap[isRNode] = range(nc, nc+nn)
            isRHEdge = isRNode[halfedge[:, 0]]
            #修改标记单元的编号
            cidxmap[halfedge[isRHEdge, 1]] = nidxmap[halfedge[isRHEdge, 0]]
            #修改未标记单元的编号
            cidxmap[~isMarkedCell[:NC]] = np.arange(nc)
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            if 'numrefine' in options:
                num0 = np.zeros(nc+nn)-10000
                num = options['numrefine']
                num[isMarkedCell[:NC]]+=1
                np.maximum.at(num0, cidxmap, num)
                options['numrefine'] = num0

            # 修改保留半边的下一个边, 上一个边
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            opp = halfedge[:, 4]
            flag1 = isMarkedHEdge[nex]
            halfedge[flag1, 2] = nex[opp[nex[flag1]]]
            flag2 = isMarkedHEdge[pre]
            halfedge[flag2, 3] = pre[opp[pre[flag2]]]

            # 对半边重新编号
            self.adjust_number(isMarkedHEdge, method='halfedge')

            #更新节点编号
            self.adjust_number(isRNode, method='node')

            # 更新halfedge
            halfedge.adjust_size(isMarkedHEdge)

            #更新起始边
            hcell.decrease_size(NC-nn-nc)
            hcell[halfedge[:, 1]] = range(len(halfedge))

            # 更新主半边
            hedge.decrease_size(isMarkedHEdge.sum()//2)
            hedge[:], = np.where(isMainHEdge[~isMarkedHEdge])

            # 更新半边层
            hlevel.adjust_size(isMarkedHEdge)

            #更新节点
            node.adjust_size(isRNode)

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2
            return isMarkedHEdge, isRNode, newNode

        elif method=='tri':
            pass

    def refine_quad(self, isMarkedCell, options={}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        halfedge = self.ds.halfedge

        if self.hedgecolor is None:
            color = 3*np.ones(NE*2, dtype = np.int_)
            color[1]=1
            while (color==3).any():
                red = color == 1
                gre = color == 0
                color[halfedge[red][:, [2,3,4]]] = 0
                color[halfedge[gre][:, [2,3,4]]] = 1
            colorlevel = ((color==1) | (color==2)).astype(np.int_)
            self.hedgecolor = {'color':color, 'level':colorlevel}

        color = self.hedgecolor['color']
        colorlevel = self.hedgecolor['level']

        node = self.entity('node')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        hedge = self.ds.hedge
        hcell = self.ds.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        isBlueHedge = color==2
        isYellowHedge = color==3

        # 得到所有单元的中心, 包括外部无界区域和区域中的洞区域
        self.ds.NV = 4
        bc = np.r_[np.zeros([cstart, 2]), node[self.ds.cell_to_node()].sum(axis = 1)/4]
        self.ds.NV = None

        nex = halfedge[isBlueHedge, 2]
        nnex = halfedge[nex, 2]
        pre = halfedge[isBlueHedge, 3]
        bc[halfedge[isBlueHedge, 1]] +=(node[halfedge[nnex, 0]] -
                node[halfedge[pre, 0]])/8# 修改蓝色半边对应单元的中心

        nex = halfedge[isYellowHedge, 2]
        nnex = halfedge[nex, 2]
        bc[halfedge[isYellowHedge, 1]] +=(node[halfedge[nex, 0]] -
                node[halfedge[isYellowHedge, 0]])/8# 修改黄色半边对应单元的中心

        # 得到标记半边并加密
        isMarkedCell[:cstart] = False
        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell, method='quad')
        isMarkedHEdge = isMarkedHEdge0 & ((color==0)|(color==1))

        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        isMarkedCell[:cstart] = False
        NN1 = self.refine_halfedge(isMarkedHEdge)

        # 改变半边的颜色
        color = np.r_[color, np.zeros(NN1*2, dtype=np.int_)]
        color[halfedge[NE*2:, 4]] = 1
        colorlevel = np.r_[colorlevel, np.zeros(NN1*2, dtype=np.int_)]
        colorlevel[halfedge[NE*2:, 4]] += 1
        NNE = NE+NN1

        # 标记的蓝色单元变成红色单元
        flag = isBlueHedge & isMarkedHEdge0
        NE1 = flag.sum()
        self.refine_cell(flag, isMarkedHEdge, method='quad_coordinateCell', options=options)

        # 修改半边颜色
        color = np.r_[color, np.zeros(NE1*2, dtype=np.int_)]
        color[NNE*2:NNE*2+NE1] = 0
        color[NNE*2+NE1:NNE*2+NE1*2] = 1
        color[halfedge[NNE*2:, 4]] = (color[NNE*2:]+1)%2
        colorlevel = np.r_[colorlevel, np.zeros(NE1*2, dtype=np.int_)]
        colorlevel[NNE*2+NE1:NNE*2+NE1*2] +=1

        #生成新单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isNewCell = (NV==6)|(NV==8)
        bc = np.r_[bc, np.zeros([NE1, 2], dtype=np.float_)]
        flag = (color==1) & isNewCell[halfedge[:, 1]]
        self.refine_cell(isNewCell, flag, method='quad', bc=bc, options=options )

        #修改半边颜色
        NC1 = flag.sum()
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NN1*2+NE1*2, dtype=np.bool_)]
        tmp = np.where(~isMarkedHEdge & flag)[0]
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[-NC1*2:-NC1] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[halfedge[tmp, 3], 4]] = 2
        colorlevel = np.r_[colorlevel, np.zeros(NC1*2, dtype = np.int_)]
        colorlevel[-NC1*2:-NC1] += 1
        self.hedgecolor['color'] = color
        self.hedgecolor['level'] = colorlevel

    def coarsen_quad(self, isMarkedCell, options={'disp': True}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.hedgecolor['color']
        colorlevel = self.hedgecolor['level']
        halfedge = self.ds.halfedge
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        # 可以移除的网格节点
        isRNode = np.ones(NN, dtype=np.bool_)
        flag = (hlevel == hlevel[halfedge[:, 4]])
        np.logical_and.at(isRNode, halfedge[:, 0], flag)
        flag = (clevel[halfedge[:, 1]]>0)
        np.logical_and.at(isRNode, halfedge[:, 0], flag)

        #粗化单元
        isMarkedHEdge, isRNode, newNode = self.coarsen_cell(isMarkedCell, isRNode)
        nn = isRNode.sum()

        #更新颜色
        color = color[~isMarkedHEdge]
        colorlevel = colorlevel[~isMarkedHEdge]

        # 标记进一步要移除的半边
        isMarkedNode = np.ones((~isRNode).sum(), dtype=np.bool_)
        pre = halfedge[:, 3]
        opp = halfedge[:, 4]
        flag0 = opp[pre[opp[pre]]] == np.arange(len(halfedge))
        np.logical_and.at(isMarkedNode, halfedge[:, 0], flag0 & (colorlevel==1))

        flag = isMarkedNode[halfedge[:, 0]] & (hlevel!=hlevel[halfedge[:, 4]])#可以变成红色半边的半边
        flag = flag[halfedge[:, 4]]
        flag[halfedge[halfedge[flag, 2], 4]] = True#要被移除的边

        #修改半边颜色
        colorlevel[halfedge[flag, 2]]-=1
        colorlevel = colorlevel[~flag]
        color = color[~flag]
        color[colorlevel==0]=0

        #删除边上的半边
        self.coarsen_halfedge(flag)

        #生成新节点, 生成新单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isNewCell = (NV==6)|(NV==8)
        bc = np.zeros([len(NV), 2], dtype=np.float_)
        bc[isNewCell] = newNode[isNewCell[-nn:]]

        tmp = (hlevel == hlevel[halfedge[:, 4]]) & (NV==6)[halfedge[:, 1]]
        tmp, = np.where(tmp)

        flag = (color==1) & (isNewCell[halfedge[:, 1]])
        NC1 = flag.sum()
        self.refine_cell(isNewCell, flag, method='quad', bc=bc, options=options)#新节点新单元

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[-NC1*2:-NC1] = 1
        color[halfedge[tmp[color[tmp]==0], 2]]=2
        color[halfedge[tmp[color[tmp]==1], 3]]=3
        color[halfedge[color==2, 4]] = 3
        color[halfedge[color==3, 4]] = 2
        colorlevel = np.r_[colorlevel, np.zeros(NC1*2, dtype = np.int_)]
        colorlevel[-NC1*2:-NC1] = 1
        self.hedgecolor['color'] = color
        self.hedgecolor['level'] = colorlevel



    def adaptive_options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=0,
            theta=1.0,
            maxsize=1e-2,
            minsize=1e-12,
            data=None,
            HB=True,
            imatrix=False,
            disp=True
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'maxsize': maxsize,
                'minsize': minsize,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options

    def adaptive(self, eta, options, method='poly'):

        if options['HB'] is True:
            HB = np.zeros((len(eta), 2), dtype=np.int)
            HB[:, 0] = np.arange(len(eta))
            HB[:, 1] = np.arange(len(eta))
            options['HB'] = HB

        NC = self.number_of_cells()
        options['numrefine'] = np.zeros(NC, dtype=np.int8)
        theta = options['theta']
        if options['method'] == 'mean':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))/1
                )
            #options['numrefine'][cellstart:] = np.around(
            #        np.log2(eta/(theta*np.mean(eta)))
            #    )
        elif options['method'] == 'max':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] == 'median':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] == 'min':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        elif options['method'] == 'numrefine':
            options['numrefine'] = eta
        elif isinstance(options['method'], float):
            val = options['method']
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/val)
                )
        else:
            raise ValueError(
                    "I don't know anyting about method %s!".format(
                        options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        isMarkedCell = (options['numrefine'] > 0)

        while np.any(isMarkedCell):
            if method=='poly':
                self.refine_poly(isMarkedCell,options=options)
            if method=='quad':
                self.refine_quad(isMarkedCell,options=options)
            if method=='nvb':
                self.refine_triangle_nvb(isMarkedCell,options=options)
            if method=='rg':
                self.refine_triangle_rg(isMarkedCell,options=options)
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            isMarkedCell = (options['numrefine'] < 0)
            while np.sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                if method=='poly':
                    self.coarsen_poly(isMarkedCell,options=options)
                if method=='quad':
                    self.coarsen_quad(isMarkedCell,options=options)
                if method=='nvb':
                    self.coarsen_triangle_nvb(isMarkedCell,options=options)
                if method=='rg':
                    self.coarsen_triangle_rg(isMarkedCell,options=options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                isMarkedCell = (options['numrefine'] < 0)

    def adaptive_refine(self, isMarkedCell, method="poly", options={}):
        if method=='nvb':
            self.refine_triangle_nvb(isMarkedCell, options=options)
        elif method=='rg':
            self.refine_triangle_rg(isMarkedCell, options=options)
        #elif method=='quad':
        #    self.refine_quad(isMarkedCell, options=options)
        elif method=='poly':
            self.refine_poly(isMarkedCell, options=options)

    def adaptive_coarsen(self, isMarkedCell, method="nvb", options={}):
        if method=='nvb':
            self.coarsen_triangle_nvb(isMarkedCell, options=options)
        elif method=='rg':
            self.coarsen_triangle_rg(isMarkedCell, options=options)
        #elif method=='quad':
        #    self.coarsen_quad(isMarkedCell, options=options)
        elif method=='poly':
            self.coarsen_poly(isMarkedCell, options=options)

    def uniform_refine(self, n=1):
        if self.ds.NV == 3:
            for i in range(n):
                self.refine_triangle_rg()
        else:
            for i in range(n):
                self.refine_poly()

    def tri_uniform_refine(self, n=1, method="rg"):
        if method == 'rg':
            for i in range(n):
                self.refine_triangle_rg()
        elif method == 'nvb':
            for i in range(n*2):
                self.refine_triangle_nvb()
        else:
            raise ValueError("refine type error! \"rg\" or \" nvb\"")


    def halfedge_direction(self):
        node = self.entity('node')
        halfedge = self.entity('halfedge')

        v = node[halfedge[:, 0]] - node[halfedge[halfedge[:, 3], 0]]
        return v

    def halfedge_length(self):
        v = self.halfedge_direction()
        l = np.linalg.norm(v, axis=1)
        return l

    def refine_marker(self, eta, theta, method="L2"):
        nc = self.number_of_all_cells()
        isMarkedCell = np.zeros(nc, dtype=np.bool_)
        isMarkedCell[self.ds.cellstart:] = mark(eta, theta, method=method)
        return isMarkedCell

    def mark_helper(self, idx):
        NC = self.number_of_cells()
        flag = np.zeros(NC, dtype=np.bool_)
        flag[idx] = True
        nc = self.number_of_all_cells()
        isMarkedCell = np.zeros(nc, dtype=np.bool_)
        isMarkedCell[self.ds.cellstart:] = flag
        return isMarkedCell

    def add_halfedge_plot(self, axes,
        index=None, showindex=False,
        nodecolor='r', edgecolor=['r', 'k'], markersize=20,
        fontsize=20, fontcolor='k', multiindex=None, linewidth=0.5):

        show_halfedge_mesh(axes, self,
                index=index, showindex=showindex,
                nodecolor=nodecolor, edgecolor=edgecolor, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor, 
                multiindex=multiindex, linewidth=linewidth)

    def split_to_trimesh(self):
        """
        @brief 在每个单元中心生成一个节点，将单元分成多个三角形单元.
                         3-------2
                         |\     /|
                         | \   / |
                         |  \ /  |
                         |   4   |
                         |  / \  |
                         | /   \ |
                         |/     \|
                         0-------1
                (0, 1, 2, 3) split to [(4, 0, 1), (4, 1, 2), (4, 2, 3), (4, 3, 0)]
        @return 分割后的三角形网格，以及每个三角形单元的第一条边对应原来多边形
                网格中的半边编号
        """
        node     = self.entity('node')
        halfedge = self.entity('halfedge')
        hcell    = self.ds.hcell

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        newnode      = np.zeros((NC+NN, node.shape[1]), dtype=np.float_)
        newnode[:NN] = node[:]
        newnode[NN:] = self.entity_barycenter('cell')

        NHE = self.ds.number_of_halfedges()
        idx, _, num = self.ds.cell_to_halfedge(returnLocalnum=True)
        newcell       = np.zeros((NHE, 3), dtype=np.int_)
        newcell[:, 2] = halfedge[idx, 0]
        newcell[:, 1] = halfedge[halfedge[idx, 3], 0]
        newcell[:, 0] = halfedge[idx, 1]+NN
        return TriangleMesh(newnode, newcell), idx, num[idx]

    def to_dual_mesh(self, projection=None):
        isbdedge = self.ds.boundary_edge_flag()
        isbdnode = self.ds.boundary_node_flag()

        NE = self.number_of_edges()
        NN = self.number_of_nodes()
        cb = self.entity_barycenter('cell')
        eb = self.entity_barycenter('edge')[isbdedge]
        nb = self.entity('node')[isbdnode]
        if projection is not None:
            eb = projection(eb)
            nb = projection(nb)
        NN0 = len(cb)
        NN1 = len(eb)
        NN2 = len(nb)

        e2newnode = -np.ones(NE, dtype=np.int_)
        e2newnode[isbdedge] = np.arange(NN0, NN0+NN1)
        h2e = self.ds.halfedge_to_edge()
        h2newnode = e2newnode[h2e]

        n2newnode = -np.ones(NN, dtype=np.int_)
        n2newnode[isbdnode] = np.arange(NN0+NN1, NN0+NN1+NN2)

        node = DynamicArray(np.r_[cb, eb, nb], dtype=self.ftype)
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        newHalfedge = DynamicArray(halfedge[:], dtype=np.int_)

        v = halfedge[:, 0]
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        newHalfedge[:, 0] = halfedge[:, 1]-cstart
        newHalfedge[:, 1] = v[opp] + cstart
        newHalfedge[:, 2] = opp[halfedge[:, 3]] 
        newHalfedge[:, 3] = nex[opp]

        h0 = np.where(halfedge[:, 1] < cstart)[0]
        h1 = opp[h0]
        newHalfedge[h0, 0] = h2newnode[h0]

        newHalfedge0 = newHalfedge.increase_size(4*NN1)
        newHalfedge[h0, 2] = np.arange(NE*2, NE*2+NN1) 
        newHalfedge[h1, 3] = np.arange(NE*2+NN1, NE*2+NN1*2)
        
        newHalfedge0[:NN1, 0] = n2newnode[v[h1]]
        newHalfedge0[:NN1, 1] = newHalfedge[h0, 1]
        newHalfedge0[:NN1, 2] = newHalfedge[opp[halfedge[h0, 3]], 3]
        newHalfedge0[:NN1, 3] = h0
        newHalfedge0[:NN1, 4] = np.arange(NE*2+NN1*2, NE*2+NN1*3)

        newHalfedge0[NN1:NN1*2, 0] = h2newnode[h0]
        newHalfedge0[NN1:NN1*2, 1] = newHalfedge[h1, 1]
        newHalfedge0[NN1:NN1*2, 2] = h1
        newHalfedge0[NN1:NN1*2, 3] = newHalfedge[nex[h0], 2]
        newHalfedge0[NN1:NN1*2, 4] = np.arange(NE*2+NN1*3, NE*2+NN1*4)

        newHalfedge0[NN1*2:NN1*3, 0] = h2newnode[h0]
        newHalfedge0[NN1*2:NN1*3, 1] = halfedge[h0, 1]
        newHalfedge0[NN1*2:NN1*3, 2] = np.arange(NE*2+NN1*3, NE*2+NN1*4)
        newHalfedge0[NN1*2:NN1*3, 3] = newHalfedge[newHalfedge[NE*2:NE*2+NN1, 2], 4]
        newHalfedge0[NN1*2:NN1*3, 4] = np.arange(NE*2, NE*2+NN1)

        newHalfedge0[NN1*3:, 0] = n2newnode[v[h0]]
        newHalfedge0[NN1*3:, 1] = halfedge[h0, 1]
        newHalfedge0[NN1*3:, 2] = newHalfedge[newHalfedge[NE*2+NN1:NE*2+NN1*2, 3], 4]
        newHalfedge0[NN1*3:, 3] = np.arange(NE*2+NN1*2, NE*2+NN1*3)
        newHalfedge0[NN1*3:, 4] = np.arange(NE*2+NN1, NE*2+NN1*2)

        newhedge = self.ds.hedge.increase_size(2*NN1)
        newhedge[:] = np.arange(NE*2, NE*2+NN1*2)

        self.ds.subdomain = DynamicArray(np.r_[self.ds.subdomain[:cstart],
            np.ones(len(node))], dtype=np.int_)

        self.ds.halfedge = newHalfedge
        self.node = node

        self.ds.NE = len(newHalfedge)//2
        self.ds.NC = NN
        self.ds.NN = len(node)

        self.ds.hcell = DynamicArray((self.ds.NC+cstart, ), dtype=self.itype) 
        self.ds.hcell[self.ds.halfedge[:, 1]] = np.arange(2*self.ds.NE) 

        self.ds.hnode = DynamicArray((self.ds.NN, ), dtype=self.itype)
        self.ds.hnode[self.ds.halfedge[:, 0]] = np.arange(2*self.ds.NE)

    def to_vtk(self, fname=None):
        import vtk
        import vtk.util.numpy_support as vnp

        NC = self.number_of_cells()

        node = self.entity('node')
        cells = self.ds.cell_to_node()

        if node.shape[1] == 2:
            node = np.c_[node, np.zeros((len(node), 1), dtype=np.float_)]
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        uGrid = vtk.vtkUnstructuredGrid()
        uGrid.SetPoints(points)

        vtk_cells = vtk.vtkCellArray()
        for c in cells:
            vtk_cell = vtk.vtkPolygon()
            vtk_cell.GetPointIds().SetNumberOfIds(len(c))
            for i, point_id in enumerate(c):
                vtk_cell.GetPointIds().SetId(i, point_id)
            vtk_cells.InsertNextCell(vtk_cell)
        uGrid.SetCells(vtk.VTK_POLYGON, vtk_cells)

        pdata = uGrid.GetPointData()
        if self.nodedata:
            nodedata = self.nodedata
            for key, val in nodedata.items():
                if val is not None:
                    val = val[:]
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool_:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])
                    d.SetName(key)
                    pdata.AddArray(d)

        if self.celldata:
            celldata = self.celldata
            cdata = uGrid.GetCellData()
            for key, val in celldata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool_:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])

                    d.SetName(key)
                    cdata.AddArray(d)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(uGrid)
        writer.Write()

    def grad_lambda(self, index=np.s_[:]):
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells() if np.array_equal(index, np.s_[:]) else len(index)
        #NC = self.number_of_cells() if index == np.s_[:] else len(index)
        v0 = node[cell[index, 2]] - node[cell[index, 1]]
        v1 = node[cell[index, 0]] - node[cell[index, 2]]
        v2 = node[cell[index, 1]] - node[cell[index, 0]]
        GD = self.geo_dimension()
        nv = np.cross(v1, v2)
        Dlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:, 0] = v0@W/length[:, None]
            Dlambda[:, 1] = v1@W/length[:, None]
            Dlambda[:, 2] = v2@W/length[:, None]
        elif GD == 3:
            length = np.linalg.norm(nv, axis=-1, keepdims=True)
            n = nv/length
            Dlambda[:, 0] = np.cross(n, v0)/length
            Dlambda[:, 1] = np.cross(n, v1)/length
            Dlambda[:, 2] = np.cross(n, v2)/length
        return Dlambda

    ## @ingroup MeshGenerators
    @classmethod
    def from_interface_cut_box_tri(cls, interface, box, nx=10, ny=10,
            keep_feature=False):
        """
        @brief 生成界面网格, 要求每个单元与界面只能交两个点或者不想交。
            步骤为:
                1. 生成笛卡尔网格
                2. 找到相交的半边
                3. 加密相交的半边
                4. 找到新生成的半边
                5. 对新生成的半边添加下一条半边或上一条半边
        @note : 1. 每个单元与界面只能交两个点或者不相交
                2. 相交的单元会被分为两个单元，界面内部的单元将继承原来单元的编号
        """
        from .interface_mesh_generator import find_cut_point, interfacemesh2d
        return cls.from_mesh(interfacemesh2d(box, interface, nx, ny))

    def cut_interface(self, interface, keep_feature=False):
        """
        @brief 生成界面网格, 要求每个单元与界面只能交两个点或者不想交。
            步骤为:
                1. 生成笛卡尔网格
                2. 找到相交的半边
                3. 加密相交的半边
                4. 找到新生成的半边
                5. 对新生成的半边添加下一条半边或上一条半边
        @note : 1. 每个单元与界面只能交两个点或者不相交
                2. 相交的单元会被分为两个单元，界面内部的单元将继承原来单元的编号
        """

        from .quadrangle_mesh import QuadrangleMesh
        from .interface_mesh_generator import find_cut_point 

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        node = self.node
        cell = self.entity('cell')
        halfedge = self.entity('halfedge')
        isMainHEdge = self.ds.main_halfedge_flag()

        phiValue = interface(node[:])
        #phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
        phiSign = np.sign(phiValue)

        # 2. 找到相交的半边
        edge = self.entity('edge')
        isCutHEdge = phiValue[halfedge[:, 0]]*phiValue[halfedge[halfedge[:, 3], 0]] < 0 

        cutHEdge, = np.where(isCutHEdge&isMainHEdge)
        cutEdge = self.ds.halfedge_to_edge(cutHEdge)

        e0 = node[edge[cutEdge, 0]]
        e1 = node[edge[cutEdge, 1]]
        cutNode = find_cut_point(interface, e0, e1)

        self.refine_halfedge(isCutHEdge, newnode = cutNode)

        NHE = self.ds.number_of_halfedges()
        newHE = np.where((halfedge[:, 0] >= NN))[0]

        if keep_feature & (hasattr(interface, "feature_points")):
            c = self.find_point(interface.feature_points)
            isCutCell = np.zeros(NC, dtype=np.bool_)
            isCutCell[halfedge[newHE, 1]] = True

            if ~np.all(isCutCell[c]):
                print("!!!!!!!! 请调整网格的重新 cut !!!!!!!!!")
            isFratrueCell = np.zeros(NC, dtype=np.bool_)
            isFratrueCell[c] = True

            fcidx = np.argsort(c)

        ## 注意这里要把 newHE 区分为界面内部和界面外部的
        cen = self.entity_barycenter('halfedge', index=newHE)
        isinnewHE = interface(cen)<0
        newHEin = newHE[isinnewHE]
        newHEout = newHE[~isinnewHE]

        idx = np.argsort(halfedge[newHEout, 1])
        newHE[::2] = newHEout[idx]
        idx = np.argsort(halfedge[newHEin, 1])
        newHE[1::2] = newHEin[idx]
        newHE = newHE.reshape(-1, 2)
        ################################################

        ne = len(newHE)
        self.number_cut_cell = ne

        halfedgeNew = halfedge.increase_size(ne*2)
        halfedgeNew[:ne, 0] = halfedge[newHE[:, 1], 0]
        halfedgeNew[:ne, 1] = halfedge[newHE[:, 1], 1]
        halfedgeNew[:ne, 2] = halfedge[newHE[:, 1], 2]
        halfedgeNew[:ne, 3] = newHE[:, 0] 
        halfedgeNew[:ne, 4] = np.arange(NHE+ne, NHE+ne*2)

        halfedgeNew[ne:, 0] = halfedge[newHE[:, 0], 0]
        halfedgeNew[ne:, 1] = np.arange(NC, NC+ne)
        halfedgeNew[ne:, 2] = halfedge[newHE[:, 0], 2]
        halfedgeNew[ne:, 3] = newHE[:, 1] 
        halfedgeNew[ne:, 4] = np.arange(NHE, NHE+ne) 

        halfedge[halfedge[newHE[:, 0], 2], 3] = np.arange(NHE+ne, NHE+ne*2)
        halfedge[halfedge[newHE[:, 1], 2], 3] = np.arange(NHE, NHE+ne)
        halfedge[newHE[:, 0], 2] = np.arange(NHE, NHE+ne)
        halfedge[newHE[:, 1], 2] = np.arange(NHE+ne, NHE+ne*2)

        isNotOK = np.ones(ne, dtype=np.bool_)
        current = np.arange(NHE+ne, NHE+ne*2)
        while np.any(isNotOK):
            halfedge[current[isNotOK], 1] = np.arange(NC, NC+ne)[isNotOK]
            current[isNotOK] = halfedge[current[isNotOK], 2]
            isNotOK = current != np.arange(NHE+ne, NHE+ne*2)
        self.ds.reinit()
        self.init_level_info()
        if keep_feature:
            flag = isFratrueCell[halfedgeNew[:ne, 1]]
            isMarkedHEdge = np.zeros(self.ds.NHE, dtype=np.bool_)
            isMarkedHEdge[np.arange(self.ds.NHE-2*ne, self.ds.NHE-ne)[flag]] = True
            isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
            self.refine_halfedge(isMarkedHEdge, newnode=interface.feature_points[fcidx])

        self.init_level_info()

    ## @ingroup MeshGenerators
    @classmethod
    def from_interface_cut_box(cls, interface, box, nx=10, ny=10,
            keep_feature=False):
        """
        @brief 生成界面网格, 要求每个单元与界面只能交两个点或者不想交。
            步骤为:
                1. 生成笛卡尔网格
                2. 找到相交的半边
                3. 加密相交的半边
                4. 找到新生成的半边
                5. 对新生成的半边添加下一条半边或上一条半边
        @note : 1. 每个单元与界面只能交两个点或者不相交
                2. 相交的单元会被分为两个单元，界面内部的单元将继承原来单元的编号
        """

        from .quadrangle_mesh import QuadrangleMesh
        from .interface_mesh_generator import find_cut_point 

        ## 1. 生成笛卡尔网格
        N = (nx+1)*(ny+1)
        NC = nx*ny
        node = np.zeros((N,2))
        X, Y = np.mgrid[box[0]:box[1]:complex(0,nx+1), box[2]:box[3]:complex(0,ny+1)]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        NN = len(node)

        idx = np.arange(N).reshape(nx+1, ny+1)
        cell = np.zeros((NC,4), dtype=np.int_)
        cell[:,0] = idx[0:-1, 0:-1].flat
        cell[:,1] = idx[1:, 0:-1].flat
        cell[:,2] = idx[1:, 1:].flat
        cell[:,3] = idx[0:-1, 1:].flat
        mesh = cls.from_mesh(QuadrangleMesh(node, cell))
        mesh.cut_interface(interface, keep_feature=keep_feature)
        return mesh

    @classmethod
    def from_box(cls, box = [0, 1, 0, 1], h=0.1):
        gmsh.initialize()
        gmsh.model.add("Box")

        # 创建单位圆
        gmsh.model.occ.addRectangle(box[0], box[2], 0.0, box[1]-box[0], box[3]-box[2])

        # 同步几何模型
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        nid2tag, node, _  = gmsh.model.mesh.get_nodes()
        node = node.reshape(-1, 3)[:, :-1]
        NN = node.shape[0]

        tag2nid = np.zeros(int(np.max(nid2tag))+1, dtype = np.int_)
        tag2nid[nid2tag] = np.arange(NN)

        cell = gmsh.model.mesh.get_elements(2, -1)[2][0].reshape(-1, 3)
        cell = tag2nid[cell]

        gmsh.finalize()

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls.from_mesh(TriangleMesh(node, cell), NV=3)

    def print(self):
        print("hcell:\n")
        for i, val in enumerate(self.ds.hcell):
            print(i, ':', val)

        print("hedge:")
        for i, val in enumerate(self.ds.hedge):
            print(i, ":", val)

        print("halfedge:")
        for i, val in enumerate(self.ds.halfedge):
            print(i, ":", val)

        print("edge:")
        edge = self.entity('edge')
        for i, val in enumerate(edge):
            print(i, ":", val)

        print("edge2cell:")
        edge2cell = self.ds.edge_to_cell()
        for i, val in enumerate(edge2cell):
            print(i, ":", val)

        cell = self.entity('cell')
        print('cell:', cell)
        cell2edge, cellLocation = self.ds.cell_to_edge()
        print("cell2edge:", cell2edge)

HalfEdgeMesh2d.set_ploter('polygon2d')

class HalfEdgeMesh2dDataStructure():
    def __init__(self, halfedge, NN=None, NC=None, NV=None):
        self.itype = halfedge.dtype
        self.halfedge = DynamicArray(halfedge, dtype=self.itype)

        self.hcell = DynamicArray((0, ), dtype=self.itype) 
        self.hnode = DynamicArray((0, ), dtype=self.itype)
        self.hedge = DynamicArray((0, ), dtype=self.itype)

        self.cell2ipt = None # cell2ipoint
        self.tree = None     # 网格节点构造的树结构

        self.reinit(NN=NN, NC=NC, NV=NV)
        self.NV = NV
        self.TD = 2

    def reinit(self, NN=None, NC=None, NV=None):
        """

        Note
        ----
        self.halfedge, 
        self.hcell: 每个单元对应的一个半边
        self.hedge: 每条边对应的一条半边有性质 hedge >= halfedge[hedge, 4]
        """

        self.cell2ipt = None # cell2ipoint
        self.tree = None     # 网格节点构造的树结构

        halfedge = self.halfedge

        self.NHE = len(halfedge) # 半边个数
        self.NBE = np.sum(halfedge[:, 4]==np.arange(self.NHE)) # 边界边个数
        self.NE = (self.NHE+self.NBE)//2
        self.NF = self.NE

        self.NN = NN if NN is not None else np.max(halfedge[:, 0])+1
        self.NC = NC if NC is not None else np.max(halfedge[:, 1])+1

        # hcell[i] is the index of one face of i-th cell
        if len(self.hcell)<self.NC:
            self.hcell.increase_size(self.NC-len(self.hcell))
        else:
            self.hcell.decrease_size(len(self.hcell)-self.NC)
        self.hcell[halfedge[:, 1]] = np.arange(self.NHE) # 的编号

        if len(self.hnode)<self.NN:
            self.hnode.increase_size(self.NN-len(self.hnode))
        else:
            self.hnode.decrease_size(len(self.hnode)-self.NN)
        self.hnode[halfedge[:, 0]] = np.arange(self.NHE)

        if len(self.hedge)<self.NE:
            self.hedge.increase_size(self.NE-len(self.hedge))
        else:
            self.hedge.decrease_size(len(self.hedge)-self.NE)
        flag = np.arange(self.NHE)-halfedge[:, 4] >= 0
        self.hedge[:] = np.arange(self.NHE)[flag]

    def top_dimension(self):
        return 2

    def number_of_edges(self):
        return self.NE 

    def number_of_cells(self):
        return self.NC 

    def number_of_halfedges(self):
        return self.NHE

    def number_of_boundary_nodes(self):
        return self.NBE

    def number_of_boundary_edges(self):
        return self.NBE

    def number_of_vertices_of_cells(self):
        NV = np.zeros(self.NC, dtype=self.itype)
        np.add.at(NV, self.halfedge[:, 1], 1)
        return NV

    def number_of_nodes_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_faces_of_cells(self):
        return self.number_of_vertices_of_cells()

    def cell_to_node(self, return_sparse=False):
        NN = self.NN
        NC = self.NC
        halfedge = self.halfedge
        if return_sparse:
            NHE = self.number_of_halfedges()
            val = np.ones(NHE, dtype=np.bool_)
            I = halfedge[:, 1]
            J = halfedge[:, 0]
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            return cell2node
        elif self.NV is None: # polygon mesh
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell.copy()
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[halfedge[current, 3], 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               current[isNotOK] = halfedge[current[isNotOK], 2]
               isNotOK = (NV0 < NV)
            return np.hsplit(cell2node, cellLocation[1:-1])
        elif self.NV == 3: # tri mesh
            cell2node = np.zeros([NC, 3], dtype = np.int_)
            current = halfedge[self.hcell[:], 2]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            return cell2node
        elif self.NV == 4: # quad mesh
            cell2node = np.zeros([NC, 4], dtype = np.int_)
            current = halfedge[self.hcell[:], 3]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 3] = halfedge[current, 0]
            return cell2node
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def cell_to_edge(self, return_sparse=False):
        NHE = self.NHE
        NE = self.NE
        NC = self.NC

        halfedge = self.halfedge
        hedge = self.hedge

        J = np.zeros(NHE, dtype=self.itype) # halfedge_to_edge
        J[hedge] = range(NE)
        J[halfedge[hedge, 4]] = range(NE)
        if return_sparse:
            val = np.ones(NHE, dtype=np.bool_)
            I = halfedge[:, 1]
            cell2edge = csr_matrix((val, (I, J)), shape=(NC, NE), dtype=np.bool_)
            return cell2edge
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)

            cell2edge = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell.copy()
            idx = cellLocation[:-1].copy()
            cell2edge[idx] = J[current]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2edge[idx[isNotOK]] = J[current[isNotOK]]
                isNotOK = (NV0 < NV)
            return cell2edge, cellLocation
        elif self.NV == 3: # tri mesh
            cell2edge = np.zeros([NC, 3], dtype = np.int_)
            current = self.hcell
            cell2edge[:, 0] = J[current]
            cell2edge[:, 1] = J[halfedge[current, 2]]
            cell2edge[:, 2] = J[halfedge[current, 3]]
            return cell2edge
        elif self.NV == 4: # quad mesh
            cell2edge = np.zeros([NC, 4], dtype=np.int_)
            current = self.hcell
            cell2edge[:, 3] = J[current]
            current = halfedge[current, 2]
            cell2edge[:, 0] = J[current] 
            current = halfedge[current, 2]
            cell2edge[:, 1] = J[current]
            current = halfedge[current, 2]
            cell2edge[:, 2] = J[current]
            return cell2edge
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def cell_to_face(self, return_sparse=True):
        return self.cell_to_edge(return_sparse=return_sparse)

    def cell_to_cell(self, return_sparse=True):
        NC = self.NC
        NHE = self.NHE
        hedge = self.hedge
        halfedge = self.halfedge

        if return_sparse:
            val = np.ones(NHE, dtype=np.bool_)
            I = halfedge[flag, 1]
            J = halfedge[halfedge[:, 4], 1]
            cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC), dtype=np.bool_)
            cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC), dtype=np.bool_)
            return cell2cell.tocsr()
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2cell = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[:].copy()
            idx = cellLocation[:-1]
            cell2cell[idx] = halfedge[halfedge[current, 4], 1]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while isNotOK.sum() > 0:
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2cell[idx[isNotOK]] = halfedge[halfedge[current[isNotOK], 4], 1]
                isNotOK = (NV0 < NV)
            idx = np.repeat(range(NC), NV)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell, cellLocation
        elif self.NV == 3: # tri mesh
            cell2cell = np.zeros((NC, 3), dtype=self.itype)
            current = self.hcell[:]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1]
            cell2cell[:, 1] = halfedge[halfedge[halfedge[current, 2], 4], 1]
            cell2cell[:, 2] = halfedge[halfedge[halfedge[current, 3], 4], 1]
            idx = np.repeat(range(NC), 3).reshape(NC, 3)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell
        elif self.NV == 4: # quad mesh
            cell2cell = np.zeros(NC, 4)
            current = self.hcell[:]
            cell2cell[:, 3] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 1] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 2] = halfedge[halfedge[current, 4], 1]
            idx = np.repeat(range(NC), 4).reshape(NC, 4)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        halfedge = self.halfedge
        hedge = self.hedge
        if return_sparse == False:
            edge = np.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[halfedge[hedge, 3], 0]
            edge[:, 1] = halfedge[hedge, 0]
            return edge
        else:
            val = np.ones(NE, dtype=np.bool_)
            edge2node = coo_matrix((val, (range(NE), halfedge[hedge, 0])),
                    shape=(NE, NN), dtype=np.bool_)
            edge2node+= coo_matrix(
                    (val, (range(NE), halfedge[halfedge[hedge, 3], 0])),
                    shape=(NE, NN), dtype=np.bool_)
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node@edge2node.tranpose()

    def edge_to_cell(self):
        """
        @brief 计算每条边对应的两个单元和边在单元中的局部编号
        """
        NE = self.NE
        NC = self.NC
        NHE = self.NHE

        halfedge = self.halfedge
        hedge = self.hedge

        J = np.zeros(NHE, dtype=self.itype) # halfedge to edge
        J[hedge] = np.arange(NE)
        J[halfedge[hedge, 4]] = np.arange(NE)

        edge2cell = np.full((NE, 4), -1, dtype=self.itype)
        edge2cell[J[hedge], 0] = halfedge[hedge, 1]
        edge2cell[J[halfedge[hedge, 4]], 1] = halfedge[halfedge[hedge, 4], 1]

        isMainHEdge = self.main_halfedge_flag() 
        if self.NV is None:
            current = self.hcell[:].copy()
            end = self.hcell[:] 
            lidx = np.zeros_like(current)
            isNotOK = np.ones_like(current, dtype=np.bool_)
            while np.any(isNotOK):
                idx = J[current[isNotOK]]
                flag = isMainHEdge[current[isNotOK]]
                edge2cell[idx[flag], 2]  = lidx[isNotOK][flag]#[:, None]
                edge2cell[idx[~flag], 3] = lidx[isNotOK][~flag]
                current[isNotOK] = halfedge[current[isNotOK], 2]
                lidx[isNotOK] += 1
                isNotOK = (current != end)
            flag = edge2cell[:, -1]==-1
            edge2cell[flag, -1] = edge2cell[flag, -2]

        elif self.NV == 3:
            current = self.hcell[:]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0

            idx = J[halfedge[current, 2]]
            flag = isMainHEdge[halfedge[current, 2]] 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            idx = J[halfedge[current, 3]]
            flag = isMainHEdge[halfedge[current, 3]] 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2
        elif self.NV == 4:
            current = self.hcell[:]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 3
            edge2cell[idx[~flag], 3] = 3
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

        flag = edge2cell[:, -1] < 0 
        edge2cell[flag, 1] = edge2cell[flag, 0]
        edge2cell[flag, 3] = edge2cell[flag, 2]
        return edge2cell

    def cell_to_edge_sign(self, return_sparse=False):
        NE = self.NE
        NC = self.NC
        NEC = self.number_of_edges_of_cells()

        edge2cell = self.edge_to_cell()
        if return_sparse == False:
            if self.NV is None:
                cellLocation = np.zeros(NC+1, dtype=self.itype)
                cellLocation[1:] = np.cumsum(NEC)
                cell2edgeSign = np.zeros(cellLocation[-1], dtype=np.bool_)
                cell2edgeSign[cellLocation[edge2cell[:, 0]] + edge2cell[:, 2]]=True
            else:
                cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
                cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True
        else:
            val = np.ones(NE, dtype=np.bool_)
            cell2edgeSign = csr_matrix(
                    (val, (edge2cell[:, 0], range(NE))),
                    shape=(NC, NE), dtype=np.bool_)
        return cell2edgeSign


    def node_to_node(self, return_sparse=True):
        NN = self.NN
        NE = self.NE
        NHE = self.NHE
        halfedge = self.halfedge
        I = halfedge[:, 0] 
        J = halfedge[halfedge[:, 3], 0] 
        val = np.ones(NHE, dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self, return_sparse=True):
        pass

    def node_to_cell(self, return_sparse=True):
        NN = self.NN
        NC = self.NC
        NHE = self.NHE
        halfedge =  self.halfedge

        if return_sparse:
            val = np.ones(NHE, dtype=np.bool_)
            I = halfedge[:, 0]
            J = halfedge[:, 1]
            node2cell = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NC), dtype=np.bool_)
            return node2cell

    def cell_to_halfedge(self, returnLocalnum=False):
        """!
        @brief 半边在所属单元中的编号
        """
        halfedge = self.halfedge

        Location = self.number_of_vertices_of_cells()
        Location = np.r_[0, np.cumsum(Location)]
        c2he = np.zeros(Location[-1], dtype=np.int_)

        NC = self.NC 
        NHE = self.NHE 
        halfedge2cellnum = np.zeros(NHE, dtype=np.int_) # 每条半边所在单元的编号
        hcell = self.hcell[:].copy()
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            c2he[Location[:-1][isNotOK]+i] = hcell[isNotOK]
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK[isNotOK] = self.hcell[:][isNotOK]!=hcell[isNotOK]
            i += 1
        if returnLocalnum:
            return c2he, Location, halfedge2cellnum
        else:
            return c2he, Location

    def halfedge_to_node_location_number(self):
        """!
        @brief 半边在所指向的顶点中的编号
        """
        N = len(self.halfedge)
        halfedge = self.halfedge
        halfedge2nodenum = np.zeros(N, dtype=np.int_) # 每条半边所在单元的编号
        hnode = self.hnode.copy()
        NN = len(hnode)
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            halfedge2nodenum[hnode[isNotOK]] = i
            hnode[isNotOK] = halfedge[hnode[isNotOK], 2]
            isNotOK[isNotOK] = self.hnode[isNotOK]!=hnode[isNotOK]
            i += 1
        return halfedge2nodenum

    def halfedge_to_cell_location_number(self):
        """!
        @brief 半边在所在单元中的编号
        """
        N = len(self.halfedge)
        halfedge = self.halfedge
        halfedge2cellnum = np.zeros(N, dtype=np.int_) # 每条半边所在单元的编号
        hcell = self.hcell.copy()
        NC = len(hcell) 
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK[isNotOK] = self.hcell[isNotOK]!=hcell[isNotOK]
            i += 1
        return halfedge2cellnum

    ######################## 与插值点有关的接口 ######################
    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief 插值点总数
        """
        if self.NV==3:
            return self.NN + (p-1)*self.NE + (p-2)*(p-1)//2*self.NC
        else:
            gdof = self.NN 
            if p > 1:
                NE = self.number_of_edges()
                NC = self.number_of_cells()
                gdof += NE*(p-1) + NC*(p-1)*p//2
            return gdof

    def number_of_local_ipoints(self,
            p: int, iptype: Union[int, str]='all') -> Union[NDArray, int]:
        """
        @brief 获取局部插值点的个数
        """
        if self.NV==3:
            if iptype in {'cell', 2}:
                return (p+1)*(p+2)//2
            elif iptype in {'face', 'edge',  1}: # 包括两个顶点
                return p + 1
            elif iptype in {'node', 0}:
                return 1
        else:
            if iptype in {'all'}:
                NV = self.number_of_vertices_of_cells()
                ldof = NV + (p-1)*NV + (p-1)*p//2
                return ldof
            elif iptype in {'cell', 2}:
                return (p-1)*p//2
            elif iptype in {'edge', 'face', 1}:
                return (p+1)
            elif iptype in {'node', 0}:
                return 1

    def cell_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief
        """
        cell = self.cell_to_node()[index]
        if p == 1:
            return cell[index]
        else:
            if self.NV==3:
                if self.cell2ipt is None:
                    mi = Mesh.multi_index_matrix(p, 2)
                    idx0, = np.nonzero(mi[:, 0] == 0)
                    idx1, = np.nonzero(mi[:, 1] == 0)
                    idx2, = np.nonzero(mi[:, 2] == 0)

                    edge2cell = self.edge_to_cell()
                    NN = self.NN
                    NE = self.number_of_edges()
                    NC = self.number_of_cells()

                    e2p = self.edge_to_ipoint(p)
                    ldof = (p+1)*(p+2)//2
                    c2p = np.zeros((NC, ldof), dtype=self.itype)

                    flag = edge2cell[:, 2] == 0
                    c2p[edge2cell[flag, 0][:, None], idx0] = e2p[flag]

                    flag = edge2cell[:, 2] == 1
                    c2p[edge2cell[flag, 0][:, None], idx1[-1::-1]] = e2p[flag]

                    flag = edge2cell[:, 2] == 2
                    c2p[edge2cell[flag, 0][:, None], idx2] = e2p[flag]


                    iflag = edge2cell[:, 0] != edge2cell[:, 1]

                    flag = iflag & (edge2cell[:, 3] == 0)
                    c2p[edge2cell[flag, 1][:, None], idx0[-1::-1]] = e2p[flag]

                    flag = iflag & (edge2cell[:, 3] == 1)
                    c2p[edge2cell[flag, 1][:, None], idx1] = e2p[flag]

                    flag = iflag & (edge2cell[:, 3] == 2)
                    c2p[edge2cell[flag, 1][:, None], idx2[-1::-1]] = e2p[flag]

                    cdof = (p-1)*(p-2)//2
                    flag = np.sum(mi > 0, axis=1) == 3
                    c2p[:, flag] = NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
                    
                    self.cell2ipt = c2p
                    return c2p[index]
                else:
                    return self.cell2ipt[index]
            else:
                NC = self.number_of_cells()
                ldof = self.number_of_local_ipoints(p, iptype='all')

                location = np.zeros(NC+1, dtype=self.itype)
                location[1:] = np.add.accumulate(ldof)

                cell2ipoint = np.zeros(location[-1], dtype=self.itype)

                edge2ipoint = self.edge_to_ipoint(p)
                edge2cell = self.edge_to_cell()

                idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
                cell2ipoint[idx] = edge2ipoint[:, 0:p]

                isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
                idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
                cell2ipoint[idx] = edge2ipoint[isInEdge, p:0:-1]

                NN = len(self.hnode)
                NV = self.number_of_vertices_of_cells()
                NE = self.number_of_edges()
                cdof = self.number_of_local_ipoints(p, iptype='cell')
                idx = (location[:-1] + NV*p).reshape(-1, 1) + np.arange(cdof)
                cell2ipoint[idx] = NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
                return np.hsplit(cell2ipoint, location[1:-1])[index]

    def edge_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief 获取网格边与插值点的对应关系

        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.NN

        edge = self.edge_to_node()[index]
        edge2ipoints = np.zeros((NE, p+1), dtype=self.itype)
        edge2ipoints[:, [0, -1]] = edge
        if p > 1:
            idx = NN + np.arange(p-1)
            edge2ipoints[:, 1:-1] =  (p-1)*index[:, None] + idx
        return edge2ipoints

    def halfedge_to_edge(self, index = np.s_[:]):
        halfedge = self.halfedge
        hedge = self.hedge
        NE = self.NE

        halfedge2edge = np.zeros(len(halfedge), dtype=np.int_)
        halfedge2edge[hedge] = np.arange(NE)
        halfedge2edge[halfedge[hedge, 4]] = np.arange(NE)
        return halfedge2edge[index] 

    def boundary_node_flag(self):
        NN = self.NN
        halfedge =  self.halfedge # DynamicArray
        isBdHEdge = halfedge[:, 4]==np.arange(self.NHE)
        isBdNode = np.zeros(NN, dtype=np.bool_)
        isBdNode[halfedge[isBdHEdge, 0]] = True 
        return isBdNode

    def nex_boundary_halfedge(self):
        halfedge = self.halfedge
        isBDHEdge = self.boundary_halfedge_flag()
        bdHEdge = np.where(isBDHEdge)[0]
        nex = 100000*np.ones(self.NHE, dtype=np.int_)
        nex[bdHEdge] = halfedge[bdHEdge, 2]

        isNotOK = np.ones(bdHEdge.shape, dtype=np.bool_)
        while np.any(isNotOK):
            nex[bdHEdge[isNotOK]] = halfedge[halfedge[nex[bdHEdge[isNotOK]], 4], 2]
            isNotOK = ~isBDHEdge[nex[bdHEdge]]
        return nex

    def boundary_halfedge_flag(self):
        return self.halfedge[:, 4]==np.arange(self.NHE)

    def boundary_edge_flag(self):
        halfedge =  self.halfedge
        hedge = self.hedge
        isBdEdge = hedge[:] == halfedge[hedge, 4] 
        return isBdEdge 

    boundary_face_flag = boundary_edge_flag

    def boundary_cell_flag(self):
        NC = self.NC
        halfedge =  self.halfedge
        isBdHEdge = halfedge[:, 4]==np.arange(self.NHE)
        isBDCell = np.zeros(NC, dtype=np.bool_)
        isBDCell[halfedge[isBdHEdge, 1]] = True 
        return isBDCell

    def boundary_node_index(self):
        NN = self.NN
        halfedge =  self.halfedge # DynamicArray
        isBdHEdge = halfedge[:, 4]==np.arange(self.NHE)
        return halfedge[isBdHEdge, 0]

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_halfedge_index(self):
        isBdHEdge = self.boundary_halfedge_flag()
        idx, = np.nonzero(isBdHEdge)
        return idx

    boundary_face_index = boundary_edge_index

    def boundary_cell_index(self):
        NN = self.NN
        halfedge =  self.halfedge # DynamicArray
        isBdHEdge = halfedge[:, 4]==np.arange(self.NHE)
        return halfedge[isBdHEdge, 1]

    def main_halfedge_flag(self):
        isMainHEdge = np.zeros(self.NHE, dtype=np.bool_)
        isMainHEdge[self.hedge[:]] = True
        return isMainHEdge

